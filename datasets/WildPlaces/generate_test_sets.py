"""
Adapted from Wild-Places repo.
"""

import os 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KDTree 
import pickle 
import argparse 
from tqdm import tqdm 
from datasets.WildPlaces.utils import load_csv, check_in_test_set
from datasets.WildPlaces.utils import P1, P2, P3, P4, P5, P6

def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name, save_dir, eval_thresh):
    database_trees = []
    test_trees = []

    # Create KD Trees
    for folder in tqdm(folders):
        # Create dataframes 
        df_database = pd.DataFrame(columns=['file', 'easting', 'northing'])
        df_test = pd.DataFrame(columns=['file', 'easting', 'northing'])

        # Load dataframe 
        df_locations = load_csv(os.path.join(base_path, runs_folder, folder, filename), os.path.join(runs_folder, folder, pointcloud_fols))
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['easting'], row['northing'], p, []) == 'test':
                df_test.loc[len(df_test)] = row 
            df_database.loc[len(df_database)] = row 

        database_tree = KDTree(df_database[['easting', 'northing']])
        test_tree = KDTree(df_test[['easting', 'northing']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in tqdm(folders):
        database = {}
        test = {}
        df_locations = load_csv(os.path.join(base_path, runs_folder, folder, filename), os.path.join(runs_folder, folder, pointcloud_fols))
        for index, row in df_locations.iterrows():
            pose = np.array(row[['x','y','z','qx','qy','qz','qw']])        
            row['timestamp'] = float(os.path.basename(row['filename'].replace('.pcd', ''))) 
            if check_in_test_set(row['easting'], row['northing'], p, []) == 'test':
                test[len(test.keys())] = {'query': row['filename'], 'northing': row['northing'], 'easting': row['easting'], 'pose': pose, 'timestamp': row['timestamp']}
            database[len(database.keys())] = {'query': row['filename'], 'northing': row['northing'],
                                              'easting': row['easting'], 'pose': pose, 'timestamp': row['timestamp']}

        database_sets.append(database)
        test_sets.append(test)
        
        # Output to file for in-run evaluation
        single_run_output_name = os.path.basename(folder) + '.pickle'
        output_to_file(database, save_dir, single_run_output_name)

    for i in tqdm(range(len(database_sets))):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["easting"], test_sets[j][key]["northing"]]])
                index = tree.query_radius(coor, r=eval_thresh)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    print(f'{output_name}: Query / Database Size {sum([len(x) for x in test_sets])} / {sum([len(x) for x in database_sets])}')

    output_to_file(database_sets, save_dir, output_name + f'_evaluation_database.pickle')
    output_to_file(test_sets, save_dir, output_name + f'_evaluation_query.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate testing dataset')
    parser.add_argument('--root', type = str, required = True, help = 'Dataset root folder')
    parser.add_argument('--save_dir', type = str, default = None, help='Directory to save training queries to, default is --root')

    parser.add_argument('--csv_filename', type = str, default = 'poses_aligned_fixed.csv', help = 'Name of CSV containing ground truth poses')
    parser.add_argument('--cloud_folder', type = str, default = 'Clouds_downsampled', help = 'Name of folder containing point cloud frames')

    parser.add_argument('--eval_thresh', type = float, default = 3, help  = 'Threshold for correct retrieval during eval')
    args = parser.parse_args()

    assert os.path.exists(args.root), f"Cannot access dataset root folder: {args.root}"
    print('Dataset root: {}'.format(args.root))

    args.save_dir = args.root if args.save_dir is None else args.save_dir
    print(f'Saving pickles to: {args.save_dir}')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # For Venman
    folders = sorted(os.listdir(os.path.join(args.root, 'Venman')))
    print('Venman')
    construct_query_and_database_sets(
        base_path = args.root,
        runs_folder = 'Venman',
        folders = folders,
        pointcloud_fols = args.cloud_folder,
        filename = args.csv_filename,
        p = [P1,P2,P3],
        output_name = 'Venman',
        save_dir = args.save_dir,
        eval_thresh = args.eval_thresh
    )

    # For Karawatha
    folders = sorted(os.listdir(os.path.join(args.root, 'Karawatha')))
    print('Karawatha')
    construct_query_and_database_sets(
        base_path = args.root,
        runs_folder = 'Karawatha',
        folders = folders,
        pointcloud_fols = args.cloud_folder,
        filename = args.csv_filename,
        p = [P4,P5,P6],
        output_name = 'Karawatha',
        save_dir = args.save_dir,
        eval_thresh = args.eval_thresh
    )

