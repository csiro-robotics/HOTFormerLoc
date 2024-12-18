"""
Visualisation of octree window partioning proposed in OctFormer.

Written by Ethan Griffiths (Data61, Pullenvale)
Adapted from OctFormer code. 
"""
import argparse 
import os
from glob import glob
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
from models.octree import OctreeT
from misc.utils import rescale_octree_points

def main():
    binPointCloudLoader = PNVPointCloudLoader()
    pcdPointCloudLoader = CSWildPlacesPointCloudLoader()
    # Get every 20th valid point cloud file
    clouds = sorted(glob(f"{args.clouds_path}/*.pcd") + glob(f"{args.clouds_path}/*.bin"))[::20]
    assert len(clouds) > 0, "No valid point cloud files found"
    
    for cloud_path in tqdm(clouds, total=len(clouds)):
        if os.path.splitext(cloud_path)[-1] == ".bin":
            points_original = binPointCloudLoader.read_pc(cloud_path)
        elif os.path.splitext(cloud_path)[-1] == ".pcd":
            points_original = pcdPointCloudLoader.read_pc(cloud_path)
        else:
            raise ValueError('Invalid point cloud type, must be .bin or .pcd')
        
        points_tensor = torch.tensor(points_original, dtype=torch.float32)
        # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
        if args.normalize or args.scale_factor is not None:
            normalize_transform = Normalize(scale_factor=args.scale_factor,
                                            unit_sphere_norm=args.unit_sphere_norm)
            points_tensor = normalize_transform(points_tensor)
        mask = torch.all(abs(points_tensor) <= 1.0, dim=1)
        points_tensor = points_tensor[mask]
        # Convert to ocnn Points object, then create Octree
        points_ocnn = Points(points_tensor)
        octree = Octree(args.max_depth, full_depth=2)
        octree.build_octree(points_ocnn)
        octree = OctreeT(  # this subclass provides patch functionality ontop of existing Octree
            octree=octree, patch_size=args.patch_size, dilation=args.dilation,
            nempty=True, max_depth=octree.depth, start_depth=octree.full_depth,
        )
        # Initialise figure        
        fig = plt.figure(figsize=(11,9))
        fig.suptitle(f"{cloud_path.split('/')[-1]}")
        # Iterate through octree depths
        for idx, depth in enumerate(range(args.max_depth, args.min_depth-1, -1)):
            # Get octree shuffled key and return points in z-order
            key = octree.key(depth, nempty=True)
            x, y, z, _ = ocnn.octree.key2xyz(key, depth)
            xyz = torch.stack([x, y, z], dim=1)
            # Convert octree point coords to original scale
            points_octree = rescale_octree_points(xyz, depth)
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
            
        plt.tight_layout()
        plt.show()

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clouds_path', type = str, required=True, help="path to processed submaps")
    parser.add_argument('--normalize', action='store_true', help="normalize submaps if they are not already in [-1, 1] range")
    parser.add_argument('--scale_factor', type = float, default=None, help="fixed scale factor to normalize by")
    parser.add_argument('--unit_sphere_norm', action='store_true', help="use unit sphere normalization")
    parser.add_argument('--max_depth', type = int, required=True, help="max depth of octree to visualise")
    parser.add_argument('--min_depth', type = int, default=2, help="min depth of octree to visualise")
    parser.add_argument('--patch_size', type = int, default=32, help="size of octree windows (# points per window)")
    parser.add_argument('--dilation', type = int, default=1, help="dilation value of octree windows")
    parser.add_argument('--cmap', type = str, default='tab20', choices=['tab10', 'tab20'], help="cmap to use")
    args = parser.parse_args()
    assert os.path.isdir(args.clouds_path), 'Invalid directory'
    assert args.max_depth >= 2, 'Octree depth must be >= 2'
    assert args.min_depth >= 2, 'Octree depth must be >= 2'
    assert args.max_depth >= args.min_depth, 'Max depth must be >= min depth'
    assert args.patch_size > 0, 'Octree patch size must be positive'
    assert args.dilation > 0, 'Dilation must be positive'
    main()