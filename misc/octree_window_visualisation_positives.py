"""
Visualisation of octree window partioning proposed in OctFormer, for positive
pairs of point clouds.

Written by Ethan Griffiths (Data61, Pullenvale)
Adapted from OctFormer code. 
"""
import argparse 
import os
import pickle
import random
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from tqdm import tqdm
import torch
import ocnn
from ocnn.octree import Octree, Points

from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from datasets.augmentation import Normalize
from datasets.coordinate_utils import CylindricalCoordinates
from models.octree import OctreeT
from misc.utils import rescale_octree_points, set_seed

BIN_PCL_LOADER = PNVPointCloudLoader()
PCD_PCL_LOADER = CSWildPlacesPointCloudLoader()
SKIP_INCREMENT = 100

def load_pcl(cloud_path: str) -> np.ndarray:
    if os.path.splitext(cloud_path)[-1] == ".bin":
        cloud = BIN_PCL_LOADER.read_pc(cloud_path)
    elif os.path.splitext(cloud_path)[-1] == ".pcd":
        cloud = PCD_PCL_LOADER.read_pc(cloud_path)
    else:
        raise ValueError('Invalid point cloud type, must be .bin or .pcd')
    return cloud    

def process_pcl(cloud: np.ndarray) -> OctreeT:    
    cloud_tensor = torch.tensor(cloud, dtype=torch.float32)
    # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
    if args.normalize or args.scale_factor is not None:
        normalize_transform = Normalize(scale_factor=args.scale_factor,
                                        unit_sphere_norm=args.unit_sphere_norm)
        cloud_tensor = normalize_transform(cloud_tensor)    
    mask = torch.all(abs(cloud_tensor) <= 1.0, dim=1)
    cloud_tensor = cloud_tensor[mask]
    # Also ensure this will hold if converting coordinate systems
    if args.coordinates == 'cylindrical':
        data_norm = torch.linalg.norm(cloud_tensor[:, :2], dim=1)[:, None]
        mask = torch.all(data_norm <= 1.0, dim=1)
        cloud_tensor = cloud_tensor[mask]
        # Convert to cylindrical coordinates
        coord_converter = CylindricalCoordinates(use_octree=True)
        cloud_tensor = coord_converter(cloud_tensor)
        
    # Convert to ocnn Points object, then create Octree
    cloud_ocnn = Points(cloud_tensor)
    octree = Octree(args.max_depth, full_depth=2)
    octree.build_octree(cloud_ocnn)
    octree = OctreeT(  # this subclass provides patch functionality ontop of existing Octree
        octree=octree, patch_size=args.patch_size, dilation=args.dilation,
        nempty=True, max_depth=octree.depth, start_depth=octree.full_depth,
        )
    return octree

def plot_octree_windows(octree: OctreeT, cloud_path: str, cloud_position: Tuple[float]):
    # Initialise figure
    fig = plt.figure(figsize=(9,7))
    fig_title = f"{cloud_path.split('/')[-1]} - coords ({cloud_position[0]:.2f}, {cloud_position[1]:.2f}) - {args.coordinates} coords"
    fig.suptitle(fig_title)
    # Iterate through octree depths
    for idx, depth in enumerate(range(args.max_depth, args.min_depth-1, -1)):
        # Get octree shuffled key and return points in z-order
        key = octree.key(depth, nempty=True)
        x, y, z, _ = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=1)
        # Convert octree point coords to original scale
        points_octree = rescale_octree_points(xyz, depth)
        # Convert coordinate system
        if args.coordinates == 'cylindrical':
            coord_converter = CylindricalCoordinates(use_octree=True)
            points_octree = coord_converter.undo_conversion(points_octree)
        # Create window partitions
        points_octree_padded = octree.patch_partition(points_octree, depth)
        patches_idx = torch.zeros(points_octree_padded.shape[0], dtype=torch.int32).view(-1, args.patch_size)
        num_windows = len(patches_idx)
        idx_values = torch.arange(num_windows, dtype=torch.int32).unsqueeze(-1)  # integer values corresponding to patch idx
        patches_idx += idx_values
        # Reverse patch operation
        if args.dilation > 1:
            patches_idx = patches_idx.view(-1, args.dilation, args.patch_size).transpose(1, 2).reshape(-1)
        else:
            patches_idx = patches_idx.reshape(-1)
        # Remove padding
        patches_idx = octree.patch_reverse(patches_idx, depth)
        # Convert to list of hex colours
        if args.cmap == 'tab20':
            patches_idx %= 20  # cap values at 20 for compatibility with tab20 colormap
            patches_idx_colours = [cm.to_hex(plt.cm.tab20(val)) for val in patches_idx]
        elif args.cmap == 'tab10':
            patches_idx %= 10  # cap values at 10 for compatibility with tab10 colormap
            patches_idx_colours = [cm.to_hex(plt.cm.tab10(val)) for val in patches_idx]
        else:
            raise ValueError(f"Unknown cmap type: {args.cmap}")
        
        # Plot the point cloud, using patches_idx for colours
        if idx > 3:
            print("[WARNING]: Plot limited to 4 depths currently! Skipping..")
            break
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        temp = ax.scatter(*points_octree.T.numpy(), c=patches_idx_colours)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"depth {depth} - {num_windows} windows")
    return None

def main():
    with open(args.training_tuples_path, 'rb') as f:
        train_tuples = pickle.load(f)

    for i in tqdm(range(0, len(train_tuples), SKIP_INCREMENT)):
        # Select anchor and positive
        anchor_path = os.path.join(args.dataset_root,
                                   train_tuples[i].rel_scan_filepath)
        anchor_position = train_tuples[i].position
        positive_path = None
        if not args.ground_aerial:
            # pick random positive
            positive_id = random.choice(train_tuples[i].positives)
            positive_path =  os.path.join(args.dataset_root,
                                          train_tuples[positive_id].rel_scan_filepath)
            positive_position = train_tuples[positive_id].position
        else:
            if 'ground' not in anchor_path:
                continue
            # Get first aerial positive
            for positive_id in train_tuples[i].positives:
                if 'ground' in train_tuples[positive_id].rel_scan_filepath:
                    continue
                positive_path = os.path.join(args.dataset_root,
                                             train_tuples[positive_id].rel_scan_filepath)
                positive_position = train_tuples[positive_id].position
                break
            if positive_path is None:
                continue

        print(f"positive distance: {np.linalg.norm(abs(anchor_position - positive_position)):.2f}m")
        
        anchor_cloud = load_pcl(anchor_path)
        positive_cloud = load_pcl(positive_path)

        anchor_octree = process_pcl(anchor_cloud)
        positive_octree = process_pcl(positive_cloud)

        plot_octree_windows(anchor_octree, anchor_path, anchor_position)
        plot_octree_windows(positive_octree, positive_path, positive_position)

        plt.tight_layout()
        plt.show()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type = str, required=True, help="root path of dataset")
    parser.add_argument('--training_tuples_path', type = str, required=True, help="path to training tuples pickle")
    parser.add_argument('--ground_aerial', action='store_true', help="Flag if visualising only ground/aerial pairs (CS-Campus3D, CS-Forest3D)")
    parser.add_argument('--normalize', action='store_true', help="normalize submaps if they are not already in [-1, 1] range")
    parser.add_argument('--coordinates', type = str, default='cartesian', choices=['cartesian','cylindrical'], help="coordinate system to use for octrees")
    parser.add_argument('--scale_factor', type = float, default=None, help="fixed scale factor to normalize by")
    parser.add_argument('--unit_sphere_norm', action='store_true', help="use unit sphere normalization")
    parser.add_argument('--max_depth', type = int, required=True, help="max depth of octree to visualise")
    parser.add_argument('--min_depth', type = int, default=2, help="min depth of octree to visualise")
    parser.add_argument('--patch_size', type = int, default=32, help="size of octree windows (# points per window)")
    parser.add_argument('--dilation', type = int, default=1, help="dilation value of octree windows")
    parser.add_argument('--cmap', type = str, default='tab20', choices=['tab10', 'tab20'], help="cmap to use")
    args = parser.parse_args()
    assert os.path.isdir(args.dataset_root), 'Invalid directory'
    assert os.path.isfile(args.training_tuples_path), 'Invalid path'
    assert args.max_depth >= 2, 'Octree depth must be >= 2'
    assert args.min_depth >= 2, 'Octree depth must be >= 2'
    assert args.max_depth >= args.min_depth, 'Max depth must be >= min depth'
    assert args.patch_size > 0, 'Octree patch size must be positive'
    assert args.dilation > 0, 'Dilation must be positive'
    # Ensure normalization type is correct for octree coordinate system
    if args.coordinates == 'cylindrical':
        if args.normalize:
            if not args.unit_sphere_norm:
                print(f"[WARNING] Unit sphere normalization recommended for cylindrical octrees")
        else:
            print(f"[WARNING] Normalization not enabled. Ensure point clouds are already normalized within unit sphere for cylindrical octrees..")
    set_seed()
    main()