"""
Script to transform WildPlaces poses.csv files into georeferenced frame, and 
post-process submaps suitable for training/testing.

By Ethan Griffiths (Data61, Pullenvale)
"""
import argparse
from os import path, listdir, makedirs

import numpy as np
import open3d as o3d
import pandas as pd

from datasets.CSWildPlaces.processing_utils import (
    quaternion_to_rot, rot_to_quaternion, CLOUD_SAVE_DIR, POSES_FILENAME,
    multiprocessing_func, make_o3d_pcl, random_down_sample, pnvlad_down_sample,
    voxel_down_sample, normalise_pcl, remove_ground_CSF,
)

RANDOM_SEED = 42

FOREST_SPLIT_DICT = {
    "Karawatha": ['K-01','K-02','K-03','K-04'],
    "Venman": ['V-01','V-02','V-03','V-04']
}

UTM_TF_DICT = {
    "K-01": np.array(
        [
            [0.99999999, 0.00124756, -0.00019022, 507176.0372],
            [-0.00124758, 0.99999992, -0.00013518, 6942599.884],
            [0.00019006, 0.00013542, 1.00000006, -7.00586056],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "K-02": np.array(
        [
            [1.00000006, -0.00065971, -0.00032731, 507168.7409],
            [0.00065967, 1.00000004, -0.00011609, 6942651.521],
            [0.00032739, 0.00011587, 1.00000002, -6.55034349],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "K-03": np.array(
        [
            [0.999998092651, -0.002285592258, 0.000356451696, 507007.0289],
            [0.002285236958, 0.999997496605, 0.000991660287, 6942887.207],
            [-0.000358713354, -0.000990847242, 0.999999582767, -2.013240337372],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "K-04": np.array(
        [
            [0.999703943729, 0.024352524430, 0.000074267300, 507148.6367],
            [-0.024352559820, 0.999703645706, 0.000593009172, 6942648.671],
            [-0.000059800164, -0.000594645855, 1.000000000000, -3.934814691544],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "V-01": np.array(
        [
            [0.99999986, -0.00041052, -0.00033836, 519786.6676],
            [0.00041029, 0.99999985, -0.00068243, 6943613.478],
            [0.00033864, 0.00068229, 0.99999967, 2.86360266],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "V-02": np.array(
        [
            [1.00000023, -0.00017173, -0.00018072, 519773.3343],
            [0.00017165, 1.00000021, -0.00046618, 6943620.908],
            [0.0001808, 0.00046616, 0.99999986, -2.4827498],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "V-03": np.array(
        [
            [0.99998626, -0.00525094, -0.00021097, 519799.4990],
            [0.0052508, 0.99998614, -0.00062337, 6943736.910],
            [0.00021424, 0.00062226, 0.99999974, -2.40304419],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "V-04": np.array(
        [
            [0.99999964, -0.00056652, -0.00016456, 519703.0248],
            [0.0005663, 0.99999884, -0.00139125, 6943668.641],
            [0.00016535, 0.00139117, 0.99999908, -5.17228534],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

SAVE_DIR_DICT = {
    "K-01": "2021_06_22_K-01_ground_sample0.5s",
    "K-02": "2021_06_21_K-02_ground_sample0.5s",
    "K-03": "2021_12_13_K-03_ground_sample0.5s",
    "K-04": "2022_08_11_K-04_ground_sample0.5s",
    "V-01": "2021_06_11_V-01_ground_sample0.5s",
    "V-02": "2021_06_11_V-02_ground_sample0.5s",
    "V-03": "2021_12_16_V-03_ground_sample0.5s",
    "V-04": "2022_08_12_V-04_ground_sample0.5s",
}

def find_nearest(cloud_timestamps_np, timestamp):
    """
    Find nearest cloud file to given timestamp.
    """
    idx = np.argmin(np.abs(cloud_timestamps_np - timestamp))
    return idx

def transform_pose(row, tf_map_to_utm) -> pd.Series:
    """
    Transform single pose value using given transform.
    """
    # Transform to local coordinates
    tf_local_to_map = quaternion_to_rot(np.array(row[['x','y','z','qx','qy','qz','qw']].astype(float)))
        
    local_to_ground = tf_map_to_utm @ tf_local_to_map
        
    # Save new pose in current row
    xyz, quat = rot_to_quaternion(local_to_ground)
    row.loc[['x','y','z','qx','qy','qz','qw']] = [*xyz, *quat]
        
    return row

def postprocess_submap(inputs):
    """
    Post-process an individual submap. Order is: Transform to UTM
    -> trim radius -> ground removal -> downsampling -> normalisation.

    Returns transformed pose on success, otherwise returns None if an error
    occured.
    """
    ii, row = inputs
    
    # transform pose
    cloud_file = CLOUD_FILES[ii]
    timestamp = row['timestamp']
    row_new = transform_pose(row, SPLIT_UTM_TF)
    
    # trim cloud
    cloud = o3d.io.read_point_cloud(path.join(CLOUD_PATH_UNFILTERED, cloud_file))
    pts = np.asarray(cloud.points)
    num_pts_before = len(pts)
    pts = pts[np.linalg.norm(pts[:,:2], axis=1) <= args.radius_max]

    # Post-processing    
    if args.remove_ground:
        pts = remove_ground_CSF(pts)
    num_pts = len(pts)
    pts_final = pts    
    if len(pts_final) < args.min_num_points:    # too few points (probably CSF's fault)
        return timestamp
    if args.downsample:
        if args.downsample_type != 'voxel' and num_pts < args.downsample_target:    # too few points after CSF
            return timestamp
        if args.downsample_type == 'random':
            pts_downsampled = random_down_sample(
                pts, args.downsample_target, RANDOM_SEED
            )
        elif args.downsample_type == 'voxel':
            pts_downsampled = voxel_down_sample(
                pts, args.voxel_size
            )
        elif args.downsample_type == 'pnvlad':
            pts_downsampled = pnvlad_down_sample(
                pts, args.downsample_target, RANDOM_SEED
            )
        else:
            raise(ValueError("Downsample type not recognised"))
        num_pts_downsampled = len(pts_downsampled)
        assert (
            args.downsample_type == 'voxel' or \
            num_pts_downsampled == args.downsample_target
        ),  f'Cloud has {num_pts_downsampled} points after downsampling'
        pts_final = pts_downsampled
    if args.normalise:
        pts_final = normalise_pcl(
            pts_final, pts, args.downsample_target, RANDOM_SEED
        )
    if len(pts_final) < args.min_num_points:    # too few points (probably CSF's fault)
        return timestamp
    
    num_pts_after = len(pts_final)
    cloud_final = make_o3d_pcl(pts_final)
    
    o3d.io.write_point_cloud(
        path.join(CLOUD_SAVE_PATH, cloud_file), cloud_final
    )
    if args.verbose:
        print(f"{ii}: {num_pts_before:>6} pts -> {num_pts_after:>6} pts", flush=True)
    
    return row_new

def main():
    """
    Read csv file of poses in (timestamp,x,y,z,qx,qy,qz,qw) format, and 
    transform to new coordinate frame using relative_transform.
    """
    global CLOUD_TIMESTAMPS, CLOUD_TIMESTAMPS_NP, CLOUD_FILES, SPLIT_UTM_TF, \
        CLOUD_SAVE_PATH, CLOUD_PATH_UNFILTERED

    # Loop through all Karawatha and Venman splits
    assert all(forest in listdir(args.root) for forest in FOREST_SPLIT_DICT.keys()), (
        "Invalid path to wild_places"
    )

    for forest, splits in FOREST_SPLIT_DICT.items():
        for split in splits:
            print(f"Processing {split}")
            split_path = path.join(args.root, forest, split)
            assert path.exists(split_path), "Invalid '--root' path, should point to 'wild_places/data'"
            split_save_dir = SAVE_DIR_DICT[split]
            save_path = path.join(args.cswildplaces_save_dir, forest, split_save_dir)
            CLOUD_PATH_UNFILTERED = path.join(split_path, args.cloud_dir)
            cloud_traj_unfiltered = path.join(split_path, args.cloud_traj)
            CLOUD_SAVE_PATH = path.join(save_path, CLOUD_SAVE_DIR)
            poses_save_path = path.join(save_path, POSES_FILENAME)
            
            if not path.exists(CLOUD_SAVE_PATH):
                makedirs(CLOUD_SAVE_PATH)
            
            SPLIT_UTM_TF = UTM_TF_DICT[split]
            
            poses = pd.read_csv(
                cloud_traj_unfiltered,
                dtype = {'timestamp':str},
                index_col = False,
            )
            poses = poses[['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]    
            CLOUD_FILES = sorted(listdir(CLOUD_PATH_UNFILTERED))

            inputs = [[ii, row] for ii, row in poses.iterrows()]
            results = multiprocessing_func(
                postprocess_submap, inputs, num_workers=args.num_workers
            )
            # Collate new poses of successfully saved submaps
            poses_temp = [x for x in results if isinstance(x, pd.Series)]               
            poses_save = pd.DataFrame(poses_temp, columns=poses.columns)
            poses_save = poses_save.sort_values(by=['timestamp'])
            poses_save.to_csv(poses_save_path, index=False)
            
            # Report removed submaps
            if len(poses_save) < len(results):
                print(f"Dropped: {len(results) - len(poses_save)} submaps")

            assert len(poses_save) == len(listdir(CLOUD_SAVE_PATH)), \
                "# of entries in poses file and # saved submaps do not match up"
            
            print(f"{split} Done\n")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True,
                        help = 'Path to wild_places/data')
    parser.add_argument('--cswildplaces_save_dir', type = str, required = True,
                        help = 'Path to CS-Wild-Places/postproc..., where the post-processed ground traversals will be saved. ENSURE YOU USE THE SAME POST-PROCESSING OPTIONS AS THE DIRECTORY YOU ARE SAVING TO.')
    parser.add_argument('--cloud_traj', type = str, default="poses_fixed.csv",
                        help = 'Trajectory file, relative to root')
    parser.add_argument('--cloud_dir', type = str, default="Clouds",
                        help = 'Clouds directory, relative to root')
    parser.add_argument('--radius_max', type = float, default = 30, 
                        help = 'Max radius (m) of submaps')
    parser.add_argument('--remove_ground', action = 'store_true', 
                        help='Remove ground points using CSF')
    parser.add_argument('--min_num_points', type = int, default = 4096,
                        help='Minimum number of points to consider as a valid submap. Useful after ground removal, incase the submap was nearly entirely flat.')
    parser.add_argument('--downsample', action = 'store_true', 
                        help='Dowsample point cloud')
    parser.add_argument('--downsample_target', type = int, default = 4096,
                        help='Number of points to downsample to')
    parser.add_argument('--downsample_type', type = str, default = 'voxel', choices = ['pnvlad', 'random', 'voxel'],
                        help='Downsampling method')
    parser.add_argument('--voxel_size', type = float, default = 0.8,
                        help='Voxel size (m), if using voxel downsampling')
    parser.add_argument('--normalise', action = 'store_true', 
                        help='Use PNVLAD normalisation to [-1, 1] range')
    parser.add_argument('--num_workers', type = int, default = 1, 
                        help = 'Number of workers for multiprocessing')
    parser.add_argument('--verbose', action = 'store_true' ,
                        help = 'Print extra info')
    args = parser.parse_args()
    print(args)
    assert (args.remove_ground or args.downsample or args.normalise), \
        "Select a post-processing option, otherwise nothing is being done!"
    if not args.downsample or args.downsample_type == 'voxel':
        args.downsample_target = None
    else:
        assert args.downsample_target >= args.min_num_points, \
            "Cannot downsample to less than minimum allowed point count."
    
    main()