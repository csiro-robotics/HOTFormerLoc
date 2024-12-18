"""
Generate train, test, and eval tuples for the CS-Wild-Places dataset. Generates 
baseline and refined set of training queries, for testing out-of-domain 
performance. 

By Ethan Griffiths (Data61, Pullenvale)
"""
from os import path, listdir, makedirs
import pickle
import random
import math
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.base_datasets import TrainingTuple

pd.options.mode.chained_assignment = None  # default='warn', disables annoying warning
CLOUD_DIR = 'clouds/'
POSES_FILE = 'poses.csv'

# Initialise random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
MARKER_SIZES = {'aerial':4, 'ground':8}

VAL_SPLITS = ['Karawatha', 'Venman']   # splits to use for validation during training
BASELINE_SPLITS = ['Karawatha', 'Venman']  # splits in baseline train set

## NEW TEST REGIONS WHICH COVER THE ENTIRE SPLITS
# For QCAT
p1 = Polygon([(490500, 6955000), (490500, 6956000), (491500, 6956000), (491500, 6955000)])
# For Samford
p2 = Polygon([(487000, 6969000), (487000, 6971000), (489000, 6971000), (489000, 6969000)])

# For Karawatha (same as wild places, but transformed to UTM frame)
p6 = Polygon([(507018.60467,6942659.3756), (507468.60473,6942659.6724),
              (507468.74853,6942441.6724), (507018.74850,6942441.3756)])
p7 = Polygon([(506953.20227,6943269.3327), (507094.20227,6943269.4257),
              (507094.33093,6943074.4257), (506953.33090,6943074.3327)])
p8 = Polygon([(506655.41198,6942951.1361), (506655.58551,6942688.1361),
              (506847.58554,6942688.2628), (506847.41204,6942951.2627)])
# For Venman (same as wild places, but transformer to UTM frame)
p9 = Polygon([(519331.85162354,6943652.20440674), (519331.19000244,6943778.20266724),
              (519485.18786621,6943779.01129150), (519494.35580444,6943747.05899048),
              (519607.18621826,6943779.65188599), (519607.84783936,6943653.65362549)])
p10 = Polygon([(519722.31359863,6943565.25347900), (519722.54461670,6943521.25408936),
               (519495.54779053,6943520.06213379), (519495.31674194,6943564.06152344)])
p11 = Polygon([(519737.04788208,6943806.33413696), (519894.04573059,6943807.15850830),
               (519941.41265869,6943737.40628052), (519940.15832520,6943595.39773560),
               (519738.16110229,6943594.33709717)])

POLY_DICT = {'QCAT':[p1], 'Samford':[p2], 'Karawatha':[p6,p7,p8], 'Venman':[p9, p10, p11]}
###

def get_timestamp_from_file(file):
    timestamp = str(path.splitext(path.split(file)[-1])[0])
    return timestamp

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def check_in_test_set(easting, northing, test_polygons, run_type, 
                      test_queries : KDTree = None):
    """
    Return which split submap is in, given polygon containing test queries. Run
    twice, first to get the ground test queries, and second to find all submaps
    in the buffer zone.
    """
    split = 'train'
    point = Point(easting, northing)
    for poly in test_polygons:
        if poly.contains(point) and run_type == 'ground':
            split = 'test'
            return split
    if test_queries is not None:
        coord = np.array([easting, northing]).reshape(1, -1)
        num_matches = test_queries.query_radius(coord, args.buffer_thresh, count_only=True)
        if num_matches[0] > 0:
            split = 'buffer'
    return split

def construct_training_query_dict(df_centroids, filename_base, test_set=False, v2_only=False):
    """
    Create training query dictionaries with v1 and v2 (MinkLoc3Dv2) formats.
    """
    run_str = 'test' if test_set else 'training'
    print(f"Computing {run_str} queries...")
    file_v1 = filename_base + 'v1.pickle'
    file_v2 = filename_base + 'v2.pickle'
    tree = KDTree(df_centroids[['easting','northing']])
    ind_pos = tree.query_radius(
        df_centroids[['easting','northing']], r=args.pos_thresh
    )
    # store ground indices to remove from test set positives (ONLY EVAL WITH GROUND QUERY, AERIAL DATABASE)
    cloud_files = df_centroids['file'].to_numpy()
    ind_ground = np.array([i for i, x in enumerate(cloud_files) if 'ground' in x])
    ind_aerial = np.array([i for i, x in enumerate(cloud_files) if 'aerial' in x])
    ind_non_neg = tree.query_radius(
        df_centroids[['easting','northing']], r=args.neg_thresh
    )
    ind_df_centroids = df_centroids.index.values.tolist()
    queries_v1 = {}
    queries_v2 = {}
    num_queries_skipped = {
        split:0 for split in POLY_DICT.keys()
    }
    count_no_positives = 0
    for anchor_ndx in tqdm(range(len(ind_pos)), desc='Computing'):
        anchor_pos = np.array(
            df_centroids.iloc[anchor_ndx][['easting', 'northing']], dtype=np.float64
        )
        query = df_centroids.iloc[anchor_ndx]['file']
        split = str.split(query, '/')[0]    # first component is split
        # Extract timestamp from the filename
        timestamp = get_timestamp_from_file(query)
        
        positives = np.setdiff1d(ind_pos[anchor_ndx], [anchor_ndx])
        negatives = np.setdiff1d(ind_df_centroids, ind_non_neg[anchor_ndx])
        non_negatives = np.sort(ind_non_neg[anchor_ndx])
        
        # remove queries with no ground positives, or remove all aerial queries if creating test set
        if (test_set and 'aerial' in query) or (
            args.query_requires_ground
            and 'aerial' in query
            and not any(
                ['ground' in file for file in df_centroids.iloc[positives]['file']]
            )
        ):
            num_queries_skipped[split] += 1
            positives = np.array([])
            negatives = np.array([])
            non_negatives = np.array([])
        # remove ground positives/negatives from test set
        elif test_set and 'ground' in query: 
            positives = np.setdiff1d(positives, ind_ground)
            negatives = np.setdiff1d(negatives, ind_ground)
            non_negatives = np.union1d(non_negatives, ind_ground)

        # remove ground/ground and aerial/aerial positives
        if args.ground_aerial_positives_only:
            if 'ground' in query:
                positives = np.setdiff1d(positives, ind_ground)
                negatives = np.setdiff1d(negatives, ind_ground)
                non_negatives = np.union1d(non_negatives, ind_ground)
            elif 'aerial' in query:
                positives = np.setdiff1d(positives, ind_aerial)
                negatives = np.setdiff1d(negatives, ind_aerial)
                non_negatives = np.union1d(non_negatives, ind_aerial)
        np.random.shuffle(negatives)

        if len(positives) == 0:
            count_no_positives += 1
        
        if not v2_only:
            queries_v1[anchor_ndx] = {
                "query":query, "positives":positives.tolist(), 
                "negatives":negatives.tolist()
            }
        queries_v2[anchor_ndx] = TrainingTuple(
            id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
            positives=positives, non_negatives=non_negatives, 
            position=anchor_pos
        )
    
    print(f"Queries with no positives: {count_no_positives}")
    print("Queries skipped per split:")
    num_queries_skipped_total = 0
    for split, num in num_queries_skipped.items():
        print(f"{split}: {num}")
        num_queries_skipped_total += num
    print(f"Final number of {run_str} queries: {len(queries_v2) - count_no_positives}/{len(queries_v2)}")
    if not v2_only:
        output_to_file(queries_v1, file_v1)
    output_to_file(queries_v2, file_v2)
    
    return True

def construct_query_and_database_sets(database_trees, database_sets, test_sets, filename_base):
    print("Saving queries and database...")
    eval_thresh = args.eval_thresh    
    file_db = filename_base + "_database.pickle"
    file_query = filename_base + "_query.pickle"
    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if(i == j):
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array(
                    [[test_sets[j][key]["easting"],
                      test_sets[j][key]["northing"]]])
                if (tree is None): # skip empty tree
                    test_sets[j][key][i] = []
                else:
                    index = tree.query_radius(coor, r=eval_thresh)
                    # indices of the positive matches in database i of each query (key) in test set j
                    test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, file_db)
    output_to_file(test_sets, file_query)
    
    return True

def format_df(df_locations: pd.DataFrame, clouds_relpath: str):
    # Fix column names
    df_locations.rename(columns={'x':'easting', 'y':'northing'}, inplace=True)
    df_locations = df_locations[['timestamp','easting','northing']]
    
    # Create filepath from timestamp
    df_locations.loc[:,'timestamp'] = clouds_relpath \
                                      + df_locations.loc[:,'timestamp'] \
                                      + '.pcd'
    df_locations.rename(columns={'timestamp':'file'}, inplace=True)
    
    return df_locations

def main():
    root_dir = args.root
    save_dir = args.save_dir
    splits = args.splits
    v2_only = args.v2_only
    
    if not path.exists(save_dir):
        makedirs(save_dir)

    # Initialize pandas DataFrame
    df_train_baseline = pd.DataFrame(columns=['file','easting','northing'])
    df_train_refined = pd.DataFrame(columns=['file','easting','northing'])
    df_test = pd.DataFrame(columns=['file','easting','northing'])
    test_queries = []

    # For visualisations
    all_coords = {}
    all_colours = {}
    all_sizes = {}

    # Find splits in dataset folder
    if len(splits) == 0:
        splits = sorted(listdir(root_dir))
        splits = [x for x in splits if path.isdir(path.join(root_dir, x))]
    for split in splits:
        print(f'Processing {split}')
        if split not in POLY_DICT.keys():
            warnings.warn('Split is not recognised, no test areas are '
                          'associated with it. Ignoring...')
            continue
        all_coords[split] = []
        all_colours[split] = []
        all_sizes[split] = []
        folders = sorted(listdir(path.join(root_dir, split)))
        print(f'Folders:\n{folders}')

        # Check folders are valid
        for folder in folders:
            assert 'ground' in folder or 'aerial' in folder, \
            f'Invalid folder "{folder}", must contain aerial or ground in name'

        # Separate ground folders
        ground_folders = [folder for folder in folders if 'ground' in folder]
        
        # Determine ground test queries
        print('Getting ground queries... ', end='')
        for folder in ground_folders:
            run_type = 'ground'
            df_locations = pd.read_csv(
                path.join(root_dir, split, folder, POSES_FILE), 
                sep=',', 
                dtype={'timestamp':str}
            )
            
            # Get easting and northing
            coords = df_locations[['x','y']].to_numpy()

            # Find ground queries
            for row in coords:
                row_split = check_in_test_set(row[0], row[1], POLY_DICT[split], 
                                            run_type, None)
                if row_split == 'test':
                    test_queries.append(row)                
        
        print('Done')
        if len(test_queries) == 0:
            print(f'WARNING: No test queries found for {split}, all will be in training set')
            test_queries_tree = None
        else:
            test_queries_tree = KDTree(test_queries)
        
        # Reset counters
        test_counter = dict.fromkeys(['aerial','ground'], 0)
        buffer_counter = dict.fromkeys(['aerial','ground'], 0)
        train_counter = dict.fromkeys(['aerial','ground'], 0)

        database_trees = []
        database_sets = []
        test_sets = []

        # Gather submaps from each folder in split        
        print('Processing submaps... ', end='')
        for folder in folders:
            df_database = pd.DataFrame(columns=['file','easting','northing'])
            database_dict = {}
            test_dict = {}
            if 'aerial' in folder:
                run_type = 'aerial' 
            elif 'ground' in folder:
                run_type = 'ground'
            else:
                raise AssertionError(f'Invalid folder "{folder}", '
                                    'must contain aerial or ground in name')
            
            df_locations = pd.read_csv(
                path.join(root_dir, split, folder, POSES_FILE),
                sep=',',
                dtype={'timestamp':str}
            )
            
            # Fix column names and filenames
            clouds_relpath = path.join(split, folder, CLOUD_DIR)
            df_locations = format_df(df_locations, clouds_relpath)
            
            # Sort submaps by train, test, and buffer set
            for _, row in tqdm(df_locations.iterrows(), desc=folder, 
                                total=len(df_locations)):
                assert path.isfile(path.join(root_dir, row['file'])), \
                    f"No associated submap for pose: {row['file']}"
                all_coords[split].append(row[['easting','northing']])
                all_sizes[split].append(MARKER_SIZES[run_type])
                row_split = check_in_test_set(row['easting'], row['northing'], 
                                              POLY_DICT[split], run_type, 
                                              test_queries_tree)
                if row_split == 'test':
                    if split in VAL_SPLITS:  # test queries only consider certain splits, for consistency with other models (as minkloc3dv2 is the only to validate using the test query tuple)
                        df_test.loc[len(df_test)] = row
                    test_dict[len(test_dict.keys())] = {
                        'query':row['file'],
                        'easting':row['easting'],
                        'northing':row['northing']
                    }
                    test_counter[run_type] += 1
                    all_colours[split].append([1,0,0])
                elif row_split == 'buffer':
                    buffer_counter[run_type] += 1
                    all_colours[split].append([1,165/255,0])
                else:
                    if split in BASELINE_SPLITS:
                        df_train_baseline.loc[len(df_train_baseline)] = row
                    df_train_refined.loc[len(df_train_refined)] = row
                    train_counter[run_type] += 1
                    all_colours[split].append([0,0,1])
                    
                if run_type == 'aerial':    # all aerial submaps form database
                    if split in VAL_SPLITS:
                        df_test.loc[len(df_test)] = row
                    df_database.loc[len(df_database)] = row
                    database_dict[len(database_dict.keys())] = {
                        'query':row['file'],
                        'easting':row['easting'],
                        'northing':row['northing']
                    }
            database_tree = KDTree(df_database[['easting','northing']]) if not df_database.empty else None
            database_trees.append(database_tree)
            database_sets.append(database_dict)
            test_sets.append(test_dict)

        print('Done')
        # save query/db pickles
        filename_base = path.join(save_dir, f"CSWildPlaces_{split}_evaluation")
        construct_query_and_database_sets(database_trees, database_sets, test_sets, filename_base)

        len_database_sets = sum([len(database_set) for database_set in database_sets])
        len_test_sets = sum([len(test_set) for test_set in test_sets])
        print(f'{split} stats:\n'
            f'\tTraining submaps - {train_counter["aerial"] + train_counter["ground"]} '
            f'({train_counter["aerial"]} aerial, {train_counter["ground"]} ground)\n'
            f'\tTest submaps     - {test_counter["aerial"] + test_counter["ground"]} '
            f'({test_counter["aerial"]} aerial, {test_counter["ground"]} ground)\n'
            f'\tBuffer submaps   - {buffer_counter["aerial"] + buffer_counter["ground"]} '
            f'({buffer_counter["aerial"]} aerial, {buffer_counter["ground"]} ground)\n'
            f'  Eval ground queries  - {len_test_sets} possible\n'
            f'  Eval aerial database - {len_database_sets}'
        )
    print(f"\nTotal number of potential baseline training submaps: "
          f"{len(df_train_baseline['file'])}")
    if args.refined:
        print(f"Total number of potential refined training submaps: "
              f"{len(df_train_refined['file'])}")
    print(f"Total number of test query submaps: {len(test_queries)}")

    ### Vis if selected ###
    if args.viz:
        all_coords_plot = {k:np.array(v) for k, v in all_coords.items()}
        split_mean = {}
        for split, split_coords in all_coords_plot.items():
            # shift to zero mean
            split_mean[split] = np.mean(split_coords, 0)
            all_coords_plot[split] = np.array(split_coords) - split_mean[split]

        fig = plt.figure(figsize=(6*len(splits)+2, 18))

        for i, split in enumerate(splits):
            ax = fig.add_subplot(2, math.ceil(len(splits)/2), i+1)
            ax.set_title(split)
            ax.set_xlabel('x [m]')
            ax.set_aspect('equal', 'box')
            if i == 0:
                ax.set_ylabel('y [m]')
            # img = plt.imread(img_dict[split]) # not currently working, needs alignment
            # ax.imshow(img)
            ax.scatter(all_coords_plot[split][:,0], all_coords_plot[split][:,1], 
                       c = all_colours[split], s = all_sizes[split])
            for poly in POLY_DICT[split]:
                xy = np.array(poly.exterior.xy) \
                     - split_mean[split].T.reshape(-1,1)
                ax.plot(*xy, 'k-')
        
        plt.tight_layout()        
        plt.show()

    if args.query_requires_ground:
        ground_positives = "_ground-positives-required_"
    elif args.ground_aerial_positives_only:
        ground_positives = "_ground-aerial-only_"
    else:
        ground_positives = "_"
    train_file_baseline_basename = path.join(save_dir, f"training_queries_CSWildPlaces_baseline{ground_positives}")
    train_file_refined_basename = path.join(save_dir, f"training_queries_CSWildPlaces_refined{ground_positives}")
    test_file_base = path.join(save_dir, "test_queries_CSWildPlaces_")
    construct_training_query_dict(df_train_baseline, train_file_baseline_basename, v2_only=v2_only)
    if args.refined:
        construct_training_query_dict(df_train_refined, train_file_refined_basename, v2_only=v2_only)
    construct_training_query_dict(df_test, test_file_base, test_set=True, v2_only=v2_only)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, 
                        help='Root directory containing splits of CS-Wild-Places dataset')
    parser.add_argument('--save_dir', type = str, default = None, 
                        help='Directory to save training queries to, default is --root')
    parser.add_argument('--splits', nargs = '+', default = [], 
                        help='Splits (min 1) in root folder to process. Processes every folder in root if empty')
    parser.add_argument('--eval_thresh', type = float, default = 15,
                        help = 'Threshold of nearest database submap for choosing eval queries')
    parser.add_argument('--pos_thresh', type = float, required = True, 
                        help = 'Threshold (m) for positive training matches, default 0.5*radius')
    parser.add_argument('--neg_thresh', type = float, required = True, 
                        help = 'Threshold (m) for negative training matches, default 2*radius')
    parser.add_argument('--buffer_thresh', type = float, required = True, 
                        help = 'Threshold (m) from ground positives to keep as buffer zone, default 1*radius')
    parser.add_argument('--query_requires_ground', default = False, action = 'store_true', 
                        help = 'Only save training queries that either are from the ground, or have at least 1 ground positive (to dissuade massive aerial bias)')
    parser.add_argument('--ground_aerial_positives_only', default = False, action = 'store_true', 
                        help = 'Only save training queries and positives that contain ground/aerial matches. Removes ground/ground or aerial/aerial positives from training.')
    parser.add_argument('--refined', default = False, action = 'store_true', 
                        help = 'Save refined training set (otherwise just baseline is saved)')
    parser.add_argument('--v2_only', default = False, action = 'store_true', 
                        help = 'Only save queries in v2 format (MinkLoc3D style). Leave as false to also generate v1 format (PNVLAD style)')
    parser.add_argument('--viz', default = False, action = 'store_true', 
                        help = 'Enable visualisations of train/test splits')
    args = parser.parse_args()

    assert not args.refined, "Refined split currently disabled!"
    if args.query_requires_ground and args.ground_aerial_positives_only:
        print("[WARNING] --ground_aerial_positives_only will supersede --query_requires_ground, thus the latter will have no effect")
        args.query_requires_ground = False
    
    args.save_dir = args.root if args.save_dir is None else args.save_dir
    
    print(args)
    main()
