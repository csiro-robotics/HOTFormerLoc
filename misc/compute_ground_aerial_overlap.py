"""
Example script for loading ground and aerial/airborne submaps, finding matched
pairs, transforming each into the correct coordinate system, then computing
overlap.
"""
import argparse 
import os

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from tqdm import tqdm

from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from datasets.CSWildPlaces.processing_utils import (
    CLOUD_SAVE_DIR, POSES_FILENAME, quaternion_to_rot, make_o3d_pcl
)
from misc.utils import set_seed, plot_points

POSITIVE_MAX_THRESH = 10  # metres, maximum threshold to consider a positive valid

def apply_transform(pc: torch.Tensor, m: torch.Tensor):
    # Apply 4x4 SE(3) transformation matrix on (N, 3) point cloud or 3x3 transformation on (N, 2) point cloud
    assert pc.ndim == 2
    n_dim = pc.shape[1]
    assert n_dim == 2 or n_dim == 3
    assert m.shape == (n_dim + 1, n_dim + 1)
    # (m @ pc.t).t = pc @ m.t
    pc = pc @ m[:n_dim, :n_dim].transpose(1, 0) + m[:n_dim, -1]
    return pc

def relative_pose(m1, m2):
    # SE(3) pose is 4x 4 matrix, such that
    # Pw = [R | T] @ [P]
    #      [0 | 1]   [1]
    # where Pw are coordinates in the world reference frame and P are coordinates in the camera frame
    # m1: coords in camera/lidar1 reference frame -> world coordinate frame
    # m2: coords in camera/lidar2 coords -> world coordinate frame
    # returns: coords in camera/lidar1 reference frame -> coords in camera/lidar2 reference frame
    #          relative pose of the first camera with respect to the second camera
    return np.linalg.inv(m2) @ m1

def load_poses_into_df(poses_csv_file: str):
    """
    Loads the poses.csv file into a pandas dataframe
    """    
    poses_df = pd.read_csv(
        poses_csv_file, 
        sep=',', 
        dtype={'timestamp':str}
    )
    return poses_df

def main():
    # Initialise point cloud loader
    pcl_loader = CSWildPlacesPointCloudLoader()
    
    # Iterate through each forest split (Kara, QCAT, etc)
    splits = sorted(os.listdir(args.postproc_path))    
    assert len(splits) > 0, "Invalid root dir, no splits found"

    for split in splits:
        print(f"Processing: {split}")
        split_path = os.path.join(args.postproc_path, split)
        # Get list of ground and aerial/airborne runs within the split
        runs = sorted(os.listdir(split_path))
        ground_run_list = [x for x in runs if "ground" in x]
        air_run_list = [x for x in runs if args.database_type in x]
        assert len(ground_run_list) > 0 and len(air_run_list) > 0, (
            f"{split_path} invalid, missing ground or {args.database_type} data"
        )
        # Should only be one aerial/airborne run per split
        assert len(air_run_list) == 1, (
            f"Should only be one {args.database_type} run per split"
        )
        air_run = air_run_list[0]
        air_run_path = os.path.join(split_path, air_run)        
        
        air_poses_df = load_poses_into_df(
            os.path.join(air_run_path, POSES_FILENAME)
        )
        print(f"  {args.database_type} run: {air_run} ({len(air_poses_df)} submaps)")
        
        # Create a KDTree of the x,y coordinates, for fast lookup of the nearest
        #   aerial/airborne submaps for each ground submap
        air_coords = air_poses_df[['x','y']].to_numpy()
        air_coords_KDTree = KDTree(air_coords)

        # Iterate through each ground run
        for ground_run in ground_run_list:
            ground_run_path = os.path.join(split_path, ground_run)
            ground_poses_df = load_poses_into_df(
                os.path.join(ground_run_path, POSES_FILENAME)
            )
            print(f"  ground run: {ground_run} ({len(ground_poses_df)} submaps)")
            ground_coords = ground_poses_df[['x','y']].to_numpy()

            # Find the closest air positive match for all ground submaps
            dists, positive_indices = air_coords_KDTree.query(ground_coords)

            # Load all positive pairs, compute relative transform, and compute chamfer dist
            for ii, ground_query_metadata in tqdm(ground_poses_df.iterrows(), total=len(ground_poses_df)):
                # Ensure there is a valid positive in range
                if dists[ii] > POSITIVE_MAX_THRESH:
                    print(f"skipping {ground_query_metadata['timestamp']} (no valid positive)")
                    continue                
                air_positive_metadata = air_poses_df.iloc[positive_indices[ii].squeeze()]
                ground_query_path = os.path.join(
                    ground_run_path, CLOUD_SAVE_DIR,
                    ground_query_metadata['timestamp'] + ".pcd"
                )
                air_positive_path = os.path.join(
                    air_run_path, CLOUD_SAVE_DIR,
                    air_positive_metadata['timestamp'] + ".pcd"
                )
                # NOTE: You may want to speedup the loop by loading and caching
                #       all submaps prior to this loop, to prevent redundant
                #       IO operations. See how the speed is though
                ground_query_pcl = torch.tensor(pcl_loader(ground_query_path))
                air_positive_pcl = torch.tensor(pcl_loader(air_positive_path))

                # Get the SE(3) poses for query and positive
                ground_query_pose = quaternion_to_rot(
                    np.array(ground_query_metadata[[
                        'x','y','z','qx','qy','qz','qw']].astype(float))
                )
                air_positive_pose = quaternion_to_rot(
                    np.array(air_positive_metadata[[
                        'x','y','z','qx','qy','qz','qw']].astype(float))
                )
                # Compute relative transform and align ground query with positive
                TF_query_to_positive = relative_pose(
                    ground_query_pose, air_positive_pose
                )
                TF_query_to_positive = torch.tensor(
                    TF_query_to_positive, dtype=torch.float
                )
                ground_query_pcl_aligned = apply_transform(
                    ground_query_pcl, TF_query_to_positive
                )

                ##############################################################
                # Compute chamfer distance here between ground_query_pcl_aligned
                # and air_positive_pcl (and compute a mean value over entire Split)
                ##############################################################
                # TODO:
                # dist = chamfer_distance(ground_query_pcl_aligned, air_positive_pcl)                

        print(f"Average chamfer distance for {split}: TO-DO")
                
    return None

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--postproc_path', type = str, required=True, help="Path to the postprocessed data to compute overlap for. E.g. '/path/CS-Wild-Places/postproc_voxel_0.80m_rmground'. ENSURE THIS IS UNNORMALISED DATA")
    parser.add_argument('--database_type', type=str, choices=['aerial', 'airborne'], help="Whether to compare ground-vs-aerial or ground-vs-airborne dist")
    args = parser.parse_args()
    assert os.path.isdir(args.postproc_path), 'Invalid path'
    print("NOTE: ENSURE THAT THE DATA IS UNNORMALISED TO ENSURE ACCURATE METRICS")
    set_seed()
    main()