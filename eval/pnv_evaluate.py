# Warsaw University of Technology

# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
# Adapted to process samples in batches by Ethan Griffiths (QUT & CSIRO Data61).

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import tqdm
import ocnn
from ocnn.octree import Octree, Points

from models.model_factory import model_factory
from misc.utils import TrainingParams, set_seed
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from datasets.augmentation import Normalize
from datasets.coordinate_utils import CylindricalCoordinates
from eval.utils import get_query_database_splits

def evaluate(model, device, params: TrainingParams, log: bool = False,
             model_name: str = 'model', show_progress: bool = False):
    # Run evaluation on all eval datasets
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    ave_recall = []
    ave_one_percent_recall = []
    ave_mrr = []
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        if 'CSWildPlaces' in params.dataset_name:
            location_name = database_file.split('_')[1]
            temp = query_file.split('_')[1]
        else:
            location_name = database_file.split('_')[0]
            temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, params, database_sets, query_sets,
                                log=log, model_name=model_name, show_progress=show_progress)
        stats[location_name] = temp
        ave_one_percent_recall.append(temp['ave_one_percent_recall'])
        ave_recall.append(temp['ave_recall'])
        ave_mrr.append(temp['ave_mrr'])
        
    # Compute average stats
    stats['average'] = {'ave_one_percent_recall': np.mean(ave_one_percent_recall),
                        'ave_recall': np.mean(ave_recall, axis=0),
                        'ave_mrr': np.mean(ave_mrr)}
    return stats


def evaluate_dataset(model, device, params: TrainingParams, database_sets, query_sets,
                     log: bool = False, model_name: str = 'model', 
                     show_progress: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []
    mrr = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    for data_set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        database_embeddings.append(get_latent_vectors(model, data_set, device, params))

    for data_set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        query_embeddings.append(get_latent_vectors(model, data_set, device, params))

    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if (i == j and params.skip_same_run) or database_embeddings[i] is None or query_embeddings[j] is None:
                continue
            pair_recall, pair_opr, pair_mrr = get_recall(i, j, database_embeddings,
                                                         query_embeddings, query_sets,
                                                         database_sets, log=log,
                                                         model_name=model_name)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            mrr.append(pair_mrr)

    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    ave_mrr = np.mean(mrr)
    stats = {'ave_one_percent_recall': ave_one_percent_recall,
             'ave_recall': ave_recall,
             'ave_mrr': ave_mrr}
    return stats


def collate_batch(data, device, params: TrainingParams):
    octrees = ocnn.octree.merge_octrees(data)
    # NOTE: remember to construct the neighbor indices
    octrees.construct_all_neigh()
    batch = {'octree': octrees.to(device)}
    return batch

def get_latent_vectors(model, data_set, device, params: TrainingParams):
    # Adapted from original PointNetVLAD code

    if params.debug:
        embeddings = np.random.rand(len(data_set), params.model_params.output_dim)
        return embeddings

    if params.dataset_name in ['Oxford','CSCampus3D']:
        pc_loader = PNVPointCloudLoader()
    elif 'CSWildPlaces' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        pc_loader = CSWildPlacesPointCloudLoader()
    else:
        raise ValueError('Invalid dataset_name')
    
    if params.normalize_points or params.scale_factor is not None:
        normalize_transform = Normalize(scale_factor=params.scale_factor,
                                        unit_sphere_norm=params.unit_sphere_norm)
        
    if params.load_octree and params.model_params.coordinates == 'cylindrical':
        coord_converter = CylindricalCoordinates(use_octree=True)

    model.eval()
    bs = params.val_batch_size
    embeddings = None
    curr_batch = []
    count_batches = 0
    for i, elem_ndx in enumerate(data_set):        
        pc_file_path = os.path.join(params.dataset_folder, data_set[elem_ndx]["query"])
        data = pc_loader(pc_file_path)
        data = torch.tensor(data)
        if params.normalize_points or params.scale_factor is not None:
            data = normalize_transform(data)
        if params.load_octree:  # Convert to Octree format
            # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
            mask = torch.all(abs(data) <= 1.0, dim=1)
            data = data[mask]
            # Also ensure this will hold if converting coordinate systems
            if params.model_params.coordinates == 'cylindrical':
                data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
                mask = torch.all(data_norm <= 1.0, dim=1)
                data = data[mask]
                # Convert to cylindrical coords
                data = coord_converter(data)
            # Convert to ocnn Points object, then create Octree
            points = Points(data)
            data = Octree(params.octree_depth, full_depth=2)
            data.build_octree(points)
        curr_batch.append(data)
        
        if len(curr_batch) >= bs or i == (len(data_set)-1):
            batch = collate_batch(curr_batch, device, params)
            embedding = compute_embedding(model, batch)
            if embeddings is None:
                embeddings = np.zeros((len(data_set), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[count_batches*bs:count_batches*bs+len(curr_batch)] = embedding
            curr_batch = []
            count_batches += 1

    return embeddings


def compute_embedding(model, batch):
    with torch.inference_mode():
        # Compute global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()
        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return embedding


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets,
               log=False, model_name: str = 'model'):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors
    recall_idx = []

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        if log:
            # Log false positives (returned as the first element)
            # Check if there's a false positive returned as the first element
            if indices[0][0] not in true_neighbors:
                fp_ndx = indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors:
                        closest_pos_ndx = indices[0][k]
                        tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[0][k]
                        tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
                                                (query_details['easting'] - tp['easting']) ** 2)
                        break
                            
                out_fp_file_name = f"{model_name}_log_fp.txt"
                with open(out_fp_file_name, "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            # Save details of 5 best matches for later visualization for 1% of queries
            s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
            for k in range(min(len(indices[0]), 5)):
                is_match = indices[0][k] in true_neighbors
                e_ndx = indices[0][k]
                e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                e_emb_dist = distances[0][k]
                world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
                                        (query_details['easting'] - e['easting']) ** 2)
                s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
            s += '\n'
            out_top5_file_name = f"{model_name}_log_search_results.txt"
            with open(out_top5_file_name, "a") as f:
                f.write(s)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                recall_idx.append(j+1)
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    mrr = np.mean(1/np.array(recall_idx))*100
    return recall, one_percent_recall, mrr


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall'],
                       stats[database_name]['ave_mrr']))
        print(stats[database_name]['ave_recall'])


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            s += f"\n[{ds}]\n"
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            ave_mrr = stats[ds]['ave_mrr']
            s += "AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, AR@N:\n".format(ave_1p_recall, ave_recall, ave_mrr)
            s += str(stats[ds]['ave_recall'])

        # NOTE: below is redundant as average is now stored in stats dict
        # mean_1p_recall = np.mean(ave_1p_recall_l)
        # mean_recall = np.mean(ave_recall_l)
        # s += "\n\nMean AR@1%: {:0.2f}, Mean AR@1: {:0.2f}\n\n".format(mean_1p_recall, mean_recall)
        s += "\n------------------------------------------------------------------------\n\n"
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--log', dest='log', action='store_true', help="Log false positives and top-5 retrievals")
    parser.set_defaults(log=False)

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('Debug mode: {}'.format(args.debug))
    print('Log search results: {}'.format(args.log))
    print('')

    set_seed()  # Seed RNG

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(params.model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)
    
    # Save results to the text file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(args.weights)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)

    stats = evaluate(model, device, params, args.log, model_name, show_progress=True)
    print_eval_stats(stats)

    pnv_write_eval_stats(f"pnv_{params.dataset_name}_results.txt", prefix, stats)

