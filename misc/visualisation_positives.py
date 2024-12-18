"""
Visualisation of positive pairs of point clouds.

Written by Ethan Griffiths (Data61, Pullenvale)
Adapted from OctFormer code. 
"""
import argparse 
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader

from misc.utils import set_seed, plot_points

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

        plot_points(anchor_cloud, show=False)
        plot_points(positive_cloud, show=False)

        plt.tight_layout()
        plt.show()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type = str, required=True, help="root path of dataset")
    parser.add_argument('--training_tuples_path', type = str, required=True, help="path to training tuples pickle")
    parser.add_argument('--ground_aerial', action='store_true', help="Flag if visualising only ground/aerial pairs (CS-Campus3D, CS-Forest3D)")
    args = parser.parse_args()
    assert os.path.isdir(args.dataset_root), 'Invalid directory'
    assert os.path.isfile(args.training_tuples_path), 'Invalid path'
    set_seed()
    main()