"""
Script to save Campus3D queries in the same format as MinkLoc3Dv2, so they can 
be tested easier on models based off PNVLAD and MinkLoc3Dv2 tuples. Only 
difference for query is that each element of the outer array should be a dict 
with integer keys, rather than an array. For training, each dict is replaced 
with a TrainingTuple class, which uses non_negatives instead of negatives to 
save space.
"""
from os import path
import pickle

import numpy as np
from tqdm import tqdm

from datasets.base_datasets import TrainingTuple


def main():
    campus_train_path = "../../data/benchmark_datasets_cs_campus3d/benchmark_datasets/training_queries_umd_4096.pickle"
    campus_query_path = "../../data/benchmark_datasets_cs_campus3d/benchmark_datasets/umd_evaluation_query.pickle"
    campus_train_fix_path = "../../data/benchmark_datasets_cs_campus3d/benchmark_datasets/training_queries_umd_4096_v2.pickle"
    campus_query_fix_path = "../../data/benchmark_datasets_cs_campus3d/benchmark_datasets/umd_evaluation_query_v2.pickle"
    
    print("Loading pickles...")
    with open(campus_train_path, 'rb') as f:
        campus_train_tuple = pickle.load(f)
    with open(campus_query_path, 'rb') as f:
        campus_query_tuple = pickle.load(f)
    print("Done")
    
    # Convert query to PNVLAD format and save
    print("Converting eval queries...")
    query_tuple_fix = []
    for i in range(len(campus_query_tuple)):
        query_tuple_fix.append({k:v for k,v in enumerate(campus_query_tuple[i])})
    with open(campus_query_fix_path, 'wb') as handle:
        pickle.dump(query_tuple_fix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done")
    
    # Convert train tuple to MinkLoc3Dv2 format
    train_queries = {}
    id_range = np.arange(len(campus_train_tuple))
    for id, item_dict in tqdm(campus_train_tuple.items(), 
                              desc='Converting training queries...'):
        timestamp = int(
            path.splitext(path.split(item_dict['query'])[-1])[0]
        )
        rel_scan_filepath = item_dict['query']
        positives = np.array(item_dict['positives'])
        non_negatives = np.setdiff1d(
            id_range, np.array(item_dict['negatives']), assume_unique=True
        )
        position = np.array([item_dict['northing'], item_dict['easting']])
        train_queries[id] = TrainingTuple(
            id=id, timestamp=timestamp, rel_scan_filepath=rel_scan_filepath,
            positives=positives, non_negatives=non_negatives, 
            position=position)
        
    with open(campus_train_fix_path, 'wb') as handle:
        pickle.dump(train_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    print("Done")


if __name__ == '__main__':
    main()