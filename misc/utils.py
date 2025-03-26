# Warsaw University of Technology

import os
import configparser
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from ocnn.octree import Octree

from datasets.coordinate_utils import CylindricalCoordinates


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        #######################################################################
        # Model dependent
        #######################################################################

        self.coordinates = params.get('coordinates', 'polar')
        assert self.coordinates in ['polar', 'cartesian', 'cylindrical'], f'Unsupported coordinates: {self.coordinates}'

        if 'cartesian' in self.coordinates:
            self.quantizer = None
        elif 'cylindrical' in self.coordinates:
            self.quantizer = CylindricalCoordinates(use_octree=True)
        else:
            raise NotImplementedError(f"Unsupported coordinates: {self.coordinates}")

        # Use cosine similarity instead of Euclidean distance
        # When Euclidean distance is used, embedding normalization is optional
        self.normalize_embeddings = params.getboolean('normalize_embeddings', False)
        self.feature_size = params.getint('feature_size', 256)
        self.pooling = params.get('pooling', 'OctGeM')
        self.num_top_down = params.getint('num_top_down', 1)

        #######################################################################
        # OctFormer params
        #######################################################################
        if 'channels' in params:  # num channels per OctFormer stage
            self.channels = tuple([int(e) for e in params['channels'].split(',')])
        else:
            self.channels = tuple([96, 192, 384, 384])
        if 'num_blocks' in params:  # num OctFormer blocks per stage
            self.num_blocks = tuple([int(e) for e in params['num_blocks'].split(',')])
        else:
            self.num_blocks = tuple([2, 2, 6, 2])  # default to OctFormer-small
        if 'num_heads' in params:  # num attention heads per stage
            self.num_heads = tuple([int(e) for e in params['num_heads'].split(',')])
        else:
            self.num_heads = None
        self.patch_size = params.getint('patch_size', 32)  # size of window attention patch
        self.dilation = params.getint('dilation', 4)  # dilation value for octree attention
        self.ct_size = params.getint('ct_size', 1)  # relay token size, if using HAT layers
        self.ct_propagation = params.getboolean('ct_propagation', False)  # propagate rt features to local features at end of stage
        self.ct_propagation_scale = params.getfloat('ct_propagation_scale', None)  # learnable scalar multiplier for rt propagation step
        self.ADaPE_mode = params.get('ADaPE_mode', None)  # Use Absolute Distribution-aware Position Encoding (ADaPE) during carrier token attention. Mode (valid values: ['pos','var','cov']) determines whether position, variance, or covariance is used (cumulative aggregation of those three)
        self.drop_path = params.getfloat('drop_path', 0.5)  # stochastic depth dropout
        self.input_features = params.get('input_features', 'P')  # P for global position, D for local displacement (check docs)
        self.downsample_input_embeddings = params.getboolean('downsample_input_embeddings', True)
        self.num_input_downsamples = params.getint('num_input_downsamples', 2)  # number of downsampling stages in ConvEmbed
        self.disable_RPE = params.getboolean('disable_RPE', False)
        self.conv_norm = params.get('conv_norm', 'batchnorm')  # choose normalisation layer after convolution layers
        assert self.conv_norm in ['batchnorm', 'layernorm', 'powernorm']
        self.layer_scale = params.getfloat('layer_scale', None)  # coefficient to initialise learnable channel-wise scalar multipliers for attention outputs, or None to disable this.
        self.grad_checkpoint = params.getboolean('grad_checkpoint', True)
        if 'qkv_init' in params:
            self.qkv_init = list([e for e in params['qkv_init'].split(',')])  # method of initialisation to use for qkv linear layers
            if len(self.qkv_init) > 1:
                self.qkv_init[1] = None if self.qkv_init[1] == 'None' else float(self.qkv_init[1])
        else:
            self.qkv_init = ['trunc_normal', 0.02]  # Second value is std dev, but is optional and can be different depening on initialisation parameters
        self.xcpe = params.getboolean('xCPE', False)  # Use xCPE instead of CPE (from PointTransformerV3)

        if 'hotformerloc' in self.model.lower():
            #######################################################################
            # HOTFormerLoc-specific params
            #######################################################################
            self.num_pyramid_levels = params.getint('num_pyramid_levels', 3)  # number of octree levels to consider for hierarchical attention.
            self.num_octf_levels = params.getint('num_octf_levels', 1)  # number of octformer levels to process local features before hierarchical attention
            self.k_pooled_tokens = params.get('k_pooled_tokens', '64')  # number of tokens to pool to when using attentional pooling
            self.disable_rt = params.getboolean('disable_rt', False)  # Disable all relay token components, and process HOTFormerLoc with solely local attention (with dilation re-enabled).
            if self.k_pooled_tokens.isdigit():
                self.k_pooled_tokens = int(self.k_pooled_tokens)
            else:
                self.k_pooled_tokens = tuple([int(e) for e in params['k_pooled_tokens'].split(',')])
        else:
            if 'ct_layers' in params:  # using carrier token attention per stage
                self.ct_layers = tuple([e == 'True' for e in params['ct_layers'].split(',')])
            else:
                self.ct_layers = tuple([False]*len(self.channels))

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e == 'quantization_step':
                s = param_dict[e]
                if self.coordinates == 'polar':
                    print(f'quantization_step - sector: {s[0]} [deg] / ring: {s[1]} [m] / z: {s[2]} [m]')
                else:
                    print(f'quantization_step: {s} [m]')
            else:
                print('{}: {}'.format(e, param_dict[e]))

        print('')


class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path: str, model_params_path: str,
                 debug: bool = False, verbose: bool = False):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug
        self.verbose = verbose

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.save_freq = params.getint('save_freq', 0)          # Model saving frequency (in epochs)
        self.eval_freq = params.getint('eval_freq', 0)          # Model eval frequency (in epochs)
        self.num_workers = params.getint('num_workers', 0)
        self.wandb = params.getboolean('wandb', True)  # enable wandb logging

        # Initial batch size for global descriptors (for both main and secondary dataset)
        self.batch_size = params.getint('batch_size', 64)
        # When batch_split_size is non-zero, multistage backpropagation is enabled
        self.batch_split_size = params.getint('batch_split_size', None)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.warmup_epochs = params.getint('warmup_epochs', None)
        self.optimizer = params.get('optimizer', 'Adam')
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                self.gamma = params.getfloat('gamma', 0.1)
                if 'scheduler_milestones' in params:
                    scheduler_milestones = params.get('scheduler_milestones')
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
                else:
                    self.scheduler_milestones = [self.epochs+1]            
            elif self.scheduler == 'ExponentialLR':
                self.gamma = params.getfloat('gamma', 0.5)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss').lower()
        if 'contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        elif self.loss == 'truncatedsmoothap':
            # Number of best positives (closest to the query) to consider
            self.positives_per_query = params.getint("positives_per_query", 4)
            # Temperatures (annealing parameter) and numbers of nearest neighbours to consider
            self.tau1 = params.getfloat('tau1', 0.01)
            self.margin = params.getfloat('margin', None)    # Margin used in loss function

        # Similarity measure: based on cosine similarity or Euclidean distance
        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)
        self.set_aug_mode = params.getint('set_aug_mode', 1)    # Augmentation mode applied to all batch samples (1 is default)
        self.random_rot_theta = params.getfloat('random_rot_theta', 5.0)    # Random rotation (in degrees) applied during training
        self.normalize_points = params.getboolean('normalize_points', False)    # Normalize points to [-1, 1]
        self.scale_factor = params.getfloat('scale_factor', None)  # Scale factor to normalize points by a fixed scale (as done in OctFormer)
        self.unit_sphere_norm = params.getboolean('unit_sphere_norm', False)  # Use unit sphere for normalization
        self.zero_mean = params.getboolean('zero_mean', True)  # Shift point cloud to zero mean during normalization
        self.octree_depth = params.getint('octree_depth', 11)    # Set depth of octree, if octrees are used
        self.full_depth = params.getint('full_depth', 2)    # Depth of octree that is fully populated
        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.validation = params.getboolean('validation', True)
        self.test_file = params.get('test_file', None)
        self.dataset_name = params.get('dataset_name', None)
        self.skip_same_run = params.getboolean('skip_same_run', True)
        self.mesa = params.getfloat('mesa', 0.0)  # MESA - memory efficient sharpness optimization, enabled if > 0.0
        self.mesa_start_ratio = params.getfloat('mesa_start_ratio', 0.25)  # when to start MESA, ratio to total training time

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)
        
        # Check if using octrees, load octrees instead of sparse tensor for OctFormer
        self.load_octree = any(model in self.model_params.model.lower() for model in ('octformer', 'hotformer'))

        # Ensure normalisation type is correct for octree coordinate system
        if self.load_octree and self.model_params.coordinates == 'cylindrical':
            if self.normalize_points:
                if not self.unit_sphere_norm:
                    print(f"[WARNING] Unit sphere normalization recommended for cylindrical octrees")
            else:
                print(f"[WARNING] Normalization not enabled. Ensure point clouds are already normalized within unit sphere for cylindrical octrees..")
        # If running a hyperparameter search
        self.hyperparam_search = params.getboolean('hyperparam_search', False)
        
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')


"""
Useful Functions
"""
def update_params_from_dict(params, param_dict: dict):    
    """
    Update training and model params from dictionary.
    """
    for key, value in param_dict.items():
        if key != 'model_params':
            setattr(params, key, value)
            continue
        for model_key, model_value in value.items():
            if model_key == 'channels_blocks_top_down_depth':
                setattr(params.model_params, 'channels', model_value[0])
                setattr(params.model_params, 'num_blocks', model_value[1])
                setattr(params.model_params, 'num_top_down', model_value[2])
                setattr(params, 'octree_depth', model_value[3])
                continue
            setattr(params.model_params, model_key, model_value)
    return params
    
def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def set_seed(seed: int = 42):
    """
    Enable (mostly) deterministic behaviour in PyTorch.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print('Determinism: Enabled')
    
def rescale_octree_points(points: torch.Tensor, depth: int) -> torch.Tensor:
    """ 
    Rescale points stored in octree to original scale.

    Args:
        points (Tensor): Points in [0, 2^d] range, where d is octree depth.
        depth (int): Octree depth used to rescale values
    """
    # rescale points to [-1, 1] since octree points are in range [0, 2^d]
    scale = 2 ** (1 - depth)
    points_scaled = points * scale - 1.0
    return points_scaled

def octree_to_points(octree: Octree, depth: int) -> torch.Tensor:
    """
    Converts averaged points in the octree to a point cloud.

    Args:
        octree (Octree): The octree to convert to a point cloud.
        depth (int): Octree depth to query points from.
        NOTE: CURRENTLY ONLY THE FINAL DEPTH CONTAINS POINTS
    """
    points = octree.points[depth]
    points_scaled = rescale_octree_points(points, depth)
    return points_scaled

def plot_points(points: np.ndarray, show=True):
    """
    Plots a point cloud using matplotlib. Colormap is based on z height.

    Args:
        points (ndarray): Point cloud of shape (N, 3), with (x,y,z) coords.
    """    
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(*points.T, c=points.T[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal', adjustable='box')
    if show:
        plt.show()

def debug_time_func(func, num_repetitions: int = 1000, inputs = (None,)):
    """Time a function's runtime with CUDA events"""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros((num_repetitions, 1))
    # GPU WARMUP
    for _ in range(10):
        _ = func(*inputs)
    # MEASURE PERFORMANCE
    for rep in range(num_repetitions):
        starter.record()
        _ = func(*inputs)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
        
    mean_syn = torch.sum(timings) / num_repetitions
    std_syn = torch.std(timings)
    print(f"{func.__class__} runtime:")
    print(f"  mean - {mean_syn:.2f}ms, std - {std_syn:.2f}ms")