"""
PyTorch util functions
"""
from typing import Union

import numpy as np
import torch

from ocnn.octree import Octree, Points


def release_cuda(x, to_numpy=False):
    """Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item, to_numpy) for item in x]
    elif isinstance(x, tuple):
        x = tuple(release_cuda(item, to_numpy) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value, to_numpy) for key, value in x.items()}
    elif isinstance(x, Octree):
        x = x.cpu()
    elif isinstance(x, Points):
        batch_size = x.batch_size  # .to() method of Points is broken, doesn't transfer batch_size property
        x = x.cpu()
        x.batch_size = batch_size
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu()
            if to_numpy:
                x = x.numpy()
    return x


def to_device(x, device: Union[torch.device, str], non_blocking=False,
              construct_octree_neigh=False):
    """Move all tensors to device."""
    if isinstance(x, list):
        x = [to_device(item, device, non_blocking, construct_octree_neigh) for item in x]
    elif isinstance(x, tuple):
        x = tuple(to_device(item, device, non_blocking, construct_octree_neigh) for item in x)
    elif isinstance(x, dict):
        x = {key: to_device(value, device, non_blocking, construct_octree_neigh) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.to(device=device, non_blocking=non_blocking)
    elif isinstance(x, Octree):
        x = x.to(device=device, non_blocking=non_blocking)
        if construct_octree_neigh:
            with torch.no_grad():
                x.construct_all_neigh()  # Ensure octree neighbours are pre-computed
    elif isinstance(x, Points):
        batch_size = x.batch_size  # .to() method of Points is broken, doesn't transfer batch_size property
        x = x.to(device=device, non_blocking=non_blocking)
        x.batch_size = batch_size
    return x
