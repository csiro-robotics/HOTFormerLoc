"""
Post-processing for CSWildPlaces submaps. Allows for downsampling,
normalisation, and removing ground points.

TODO List:
    - Disable CSF console output (low prio)

By Ethan Griffiths (Data61, CSIRO)
"""
from os import path, makedirs, listdir
import argparse
import glob
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import open3d as o3d

from datasets.CSWildPlaces.processing_utils import random_down_sample, pnvlad_down_sample, voxel_down_sample, \
    normalise_pcl, remove_ground_CSF, make_o3d_pcl, multiprocessing_func, \
    CLOUD_SAVE_DIR, POSES_FILENAME

RANDOM_SEED = 42
SAVE_FOLDER_BASE = 'postproc'

def save_info(root: str, save_dir: str):
    """
    Save txt file with info of what post-processing was done to what data.
    """
    txt_file = path.join(save_dir, 'postproc_info.txt')
    with open(txt_file, 'w') as f:
        now = datetime.now()
        f.write(f'Created: {now.strftime("%Y/%m/%d-%H:%M:%S")}\n\n')
        f.write(f'Root folder: {path.abspath(root)}\n')
        f.write('Args:\n')
        f.write(str(args))
    return True

def postprocess_submap(submap: str):
    """
    Post-process an individual submap. Order is: ground removal -> downsampling 
    -> normalisation.
    """
    timestamp = path.splitext(path.split(submap)[1])[0]
    if args.debug:
        print(timestamp)
    cloud = o3d.io.read_point_cloud(submap)
    pts = np.asarray(cloud.points)
    if args.remove_ground:
        pts = remove_ground_CSF(pts, args.debug)
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
    cloud_final = make_o3d_pcl(pts_final)
    o3d.io.write_point_cloud(   # Save downsampled cloud
        path.join(SAVE_DIR, path.relpath(submap, args.root)), cloud_final
    )
    
    return None

def postprocessing():
    global SAVE_DIR
    SAVE_DIR = args.save_dir
    if SAVE_DIR is None:
        save_folder = SAVE_FOLDER_BASE
        if args.downsample:
            downsample_str = 'rand' if args.downsample_type == 'random' else args.downsample_type
            if args.downsample_type == 'voxel':
                save_folder += f'_{downsample_str}_ds_{args.voxel_size:0.2f}m'
            else:
                save_folder += f'_{downsample_str}_ds_{args.downsample_target}'
        if args.remove_ground:
            save_folder += '_rmground'
        if args.normalise:
            save_folder += '_normalised'
        SAVE_DIR = path.join(args.root, f'../{save_folder}/')
    
    if path.exists(SAVE_DIR):
        print(f"[WARNING] Save directory '{SAVE_DIR}' already exists. Overwriting in 5 seconds...")
        sleep(5)
    else:
        makedirs(SAVE_DIR)
    
    _ = save_info(args.root, SAVE_DIR)
    
    # Iterate through each split
    splits = args.splits
    if splits == []:
        splits = sorted(listdir(args.root))
    
    assert len(splits) > 0, 'Invalid root dir, no splits found'

    for split in splits:
        split_path = path.join(args.root, split)
        if not path.isdir(split_path):
            continue
        
        for folder in sorted(glob.glob(f'{split_path}/*/')):
            if any([dir in folder for dir in args.exclude_dirs]):
                print(f"Skipping '{folder}'")
                continue
            folder_relpath = path.relpath(folder, args.root)
            folder_save_dir = path.join(SAVE_DIR, folder_relpath, CLOUD_SAVE_DIR)
            poses_save_path = path.join(SAVE_DIR, folder_relpath, POSES_FILENAME)
            if not path.exists(folder_save_dir):
                makedirs(folder_save_dir)
            
            # Multiprocessing
            print(f"Processing '{folder}'")
            inputs = glob.glob(f'{folder}/**/*.pcd')
            results = multiprocessing_func(
                postprocess_submap, inputs, num_workers=args.num_workers
            )
            failed_submaps = [x for x in results if x is not None]
            
            # Copy poses
            poses_path = path.join(folder, POSES_FILENAME)
            poses = pd.read_csv(poses_path, dtype={'timestamp':str})
            # filter out removed submaps
            if len(failed_submaps) > 0:
                print(f"Dropped: {len(failed_submaps)} submaps")
                poses = poses[~poses.timestamp.str.contains('|'.join(failed_submaps))]
            assert len(poses) == len(listdir(folder_save_dir)), \
                "# of entries in poses file and # saved submaps do not match up"
            poses.to_csv(poses_save_path, index=False)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, 
                        help='Root directory containing split folders of CSWildPlaces dataset')
    parser.add_argument('--save_dir', type = str, default = None, 
                        help='Directory to save downsampled splits to, default is the parent directory of --root')
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
                        help='Enable multiprocessing, specifying number of workers')
    parser.add_argument('--splits', nargs = '+', default = [], 
                        help='Splits (min 1) in root folder to process. Processes every folder in root if empty.')
    parser.add_argument('--exclude_dirs', nargs = '+', default = [], 
                        help='List of dirs to ignore during preprocessing')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging messages and visualisations')
    args = parser.parse_args()
    print(args)
    assert (args.remove_ground or args.downsample or args.normalise), \
        "Select a post-processing option, otherwise nothing is being done!"
    if not args.downsample or args.downsample_type == 'voxel':
        args.downsample_target = None
    else:
        assert args.downsample_target >= args.min_num_points, \
            "Cannot downsample to less than minimum allowed point count."
    
    postprocessing()
