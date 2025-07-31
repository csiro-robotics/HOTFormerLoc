"""
Scripts to visualise output embeddings for anchor, pos, negs.
Ethan Griffiths (QUT & CSIRO Data61).
"""

import numpy as np
import pickle
import os
import argparse
import torch
import random
from tqdm import tqdm
import ocnn
from ocnn.octree import Octree, Points
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as cm

from models.model_factory import model_factory
from misc.torch_utils import to_device, release_cuda
from misc.utils import TrainingParams, set_seed
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader
from datasets.augmentation import Normalize
from datasets.coordinate_utils import CylindricalCoordinates
from eval.utils import get_query_database_splits


def query_distance(query, query_list: list[dict]) -> float:
    """
    Returns the distance of closest query in list.
    """
    distance = 1e10
    if len(query_list) > 0:
        query_positions = np.array([[q['northing'], q['easting']] for q in query_list])
        kdtree = KDTree(query_positions, leaf_size=2)
        query_position = np.array([query['northing'], query['easting']]).reshape(1, -1)
        all_distances, _ = kdtree.query(query_position, k=1)
        distance = all_distances[0]
    return distance

def get_cumulative_indices(samples_list) -> list[int]:
    """
    Return cumulative indices of samples in list, to recover them after flattening.
    """
    cumulative_indices_list = []
    cumulative_length = 0
    for samples in samples_list[:-1]:  # ignore last list, as np.split infers its length
        cumulative_length += len(samples)
        cumulative_indices_list.append(cumulative_length)
    return cumulative_indices_list

def visualise_embeddings(model, device, num_queries: int, query_min_distance: float, params: TrainingParams):
    model.eval()  # Eval mode
    # Get queries and database for desired split
    eval_database_files, eval_query_files = get_query_database_splits(params)

    assert len(eval_database_files) == len(eval_query_files)

    # Just use default split for visualisation (Oxford - Oxford, CSWildPlaces - Karawatha)
    database_file = eval_database_files[0]
    query_file = eval_query_files[0]
    
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

    # Sample queries and collect positives (ensuring queries are far apart)
    query_list = []
    for _ in tqdm(range(num_queries), desc='Selecting query set'):
        while len(query_sets) > 0:  # loop until valid query is chosen
            set_idx = random.randint(0,len(query_sets)-1)
            if len(query_sets[set_idx]) == 0:
                query_sets.pop(set_idx)
                continue
            rand_idx = random.choice(list(query_sets[set_idx].keys()))
            query_details = query_sets[set_idx].pop(rand_idx)  # ensure it is not selected again
            if query_distance(query_details, query_list) >= query_min_distance:
                query_list.append(query_details)
                break
        if len(query_sets) == 0:
            print(f"[WARNING] No more possible queries with current distance threshold, continuing with {len(query_list)} queries")
            break
    samples_list = [[query['query']] for query in query_list]  # first elem in each is anchor

    # Collect positives for each query
    for idx, query_details in enumerate(query_list):
        for jdx, data_set in enumerate(database_sets):
            if jdx in query_details.keys():
                for positive_idx in query_details[jdx]:
                    samples_list[idx].append(data_set[positive_idx]['query'])

    # Flatten samples list and get embeddings from model
    samples_flattened_list = [elem for sample in samples_list for elem in sample]
    cumulative_idx_list = get_cumulative_indices(samples_list)  # used for undoing flattening
    embeddings = get_latent_vectors(model, samples_flattened_list, device, params)

    # Visualise embedding space with TSNE, colorised by query/positive group
    print('Computing TSNE...', end='', flush=True)
    tsne = TSNE(random_state=42, n_iter=2000, metric='cosine', perplexity=30)  # try euclidean and cosine
    embeddings_tsne = tsne.fit_transform(embeddings)
    print(' Done!')
    embeddings_tsne_list = np.split(embeddings_tsne, cumulative_idx_list)

    # Get color list and plot
    if num_queries > 20:
        print(f"[WARNING] Using tab20 colormap, which is incompatible with more than 20 queries")
    colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]
    # colors = [cm.to_hex(plt.cm.plasma(i)) for i in np.linspace(0,1,20)]
    fig, axs = plt.subplots(1,1)
    marker_size = 35
    for idx, sample_embeddings in enumerate(embeddings_tsne_list):
        if idx == 0:  # plot anchor first just to make legend look nicer
            axs.scatter(sample_embeddings[0,0], sample_embeddings[0,1], s=marker_size*2, c=colors[idx], marker='*', edgecolors='black')
        axs.scatter(sample_embeddings[1:,0], sample_embeddings[1:,1], s=marker_size, c=colors[idx], alpha=0.3)
        axs.scatter(sample_embeddings[0,0], sample_embeddings[0,1], s=marker_size*2, c=colors[idx], marker='*', edgecolors='black')  # identify anchor embedding
        if idx == 0:
            axs.legend(['Anchor', 'Positives'])
    axs.set_title(f'TSNE Projection of {params.model_params.model} embeddings on {params.dataset_name} dataset')
    plt.show()
    
    return None

def collate_batch(data, device, params: TrainingParams):
    octrees = ocnn.octree.merge_octrees(data)
    # NOTE: remember to construct the neighbor indices
    batch = to_device({'octree': octrees}, device, construct_octree_neigh=True)
    return batch

def get_latent_vectors(model, data_list: list[str], device, params: TrainingParams) -> np.ndarray:
    if params.dataset_name in ['Oxford','CSCampus3D']:
        pc_loader = PNVPointCloudLoader()
    elif 'CSWildPlaces' in params.dataset_name or 'WildPlaces' in params.dataset_name:
        pc_loader = CSWildPlacesPointCloudLoader()

    bs = params.val_batch_size
    embeddings = None
    curr_batch = []
    count_batches = 0
    for i, pc_rel_path in tqdm(enumerate(data_list), desc='Computing embeddings', total=len(data_list)):
        pc_abs_path = os.path.join(params.dataset_folder, pc_rel_path)
        data = pc_loader(pc_abs_path)
        data = torch.tensor(data)
        if params.normalize_points or params.scale_factor is not None:
            data = Normalize(scale_factor=params.scale_factor,
                             unit_sphere_norm=params.unit_sphere_norm)(data)
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
                coord_converter = CylindricalCoordinates(use_octree=True)
                data = coord_converter(data)                
            # Convert to ocnn Points object, then create Octree
            points = Points(data)
            data = Octree(params.octree_depth, full_depth=2)
            data.build_octree(points)
        curr_batch.append(data)
        
        if len(curr_batch) >= bs or i == (len(data_list)-1):
            batch = collate_batch(curr_batch, device, params)
            embedding = compute_embedding(model, batch)
            if embeddings is None:
                embeddings = np.zeros((len(data_list), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[count_batches*bs:count_batches*bs+len(curr_batch)] = embedding
            curr_batch = []
            count_batches += 1

    return embeddings


def compute_embedding(model, batch):
    with torch.inference_mode():
        # Compute global descriptor
        y = model(batch)
        embedding = release_cuda(y['global'], to_numpy=True)
    return embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise embeddings for positives and negatives.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--num_queries', type=int, default=20, help='Number of queries to sample')
    parser.add_argument('--query_min_distance', type=float, default=100, help='Minimum distance of separation between queries')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
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
        if os.path.splitext(args.weights)[1] == '.ckpt':
            state = torch.load(args.weights)
            model.load_state_dict(state['model_state_dict'])
        else:  # .pt or .pth
            model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    visualise_embeddings(model, device, args.num_queries, args.query_min_distance, params)

