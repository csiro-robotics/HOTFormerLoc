import math
import random
from typing import Optional

import numpy as np
import torch
from scipy.linalg import expm, norm
from torchvision import transforms as transforms


class TrainSetTransform:
    def __init__(self, aug_mode, random_rot_theta: float = 5.0):
        self.aug_mode = aug_mode
        self.transform = None
        if self.aug_mode == 1:
            t = [RandomRotation(max_theta=random_rot_theta, axis=np.array([0, 0, 1])),
                 RandomFlip([0.25, 0.25, 0.])]
        elif self.aug_mode == 2:
            t = [RandomFlip([0.25, 0.25, 0.])]
        elif self.aug_mode == 0:    # No augmentations
            return None
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=None):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180.) * 2. * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180.) * 2. * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords

class Normalize:
    """
    Normalize cloud within `[-norm_range, norm_range]`. Defaults to `[-1, 1]`.

    Alternatively, provide `scale_factor` to normalize the cloud with a fixed
    scaling factor. E.g. `cloud_normalized = (cloud - centroid) / scale_factor`.
    Also supports normlization within a unit sphere, and normalizing with and
    without shifting to zero mean.
    """    
    def __init__(self, norm_range: Optional[float] = None,
                 scale_factor: Optional[float] = None,
                 unit_sphere_norm: bool = False,
                 zero_mean: bool = True):
        assert not all([arg is not None for arg in [norm_range, scale_factor]]),\
            "Must specify one of norm_range or scale_factor, not both"
        self.norm_range = 1.0
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        if norm_range is not None:
            assert norm_range > 0, "Range must be positive"
            self.norm_range = norm_range
        elif scale_factor is not None:
            assert scale_factor > 0, "Scale factor must be positive"
            self.norm_range = None
            self.scale_factor = scale_factor

    def __call__(self, coords: torch.Tensor):
        if not self.unit_sphere_norm:                
            bbmin = coords.min(dim=0).values
            bbmax = coords.max(dim=0).values
            if self.zero_mean:
                center = (bbmin + bbmax) * 0.5
                coords = (coords - center)
            if self.scale_factor is not None:
                coords_normalized = coords / self.scale_factor
            else:
                box_size = (bbmax - bbmin).max() + 1.0e-6
                coords_normalized = coords * (2.0 * self.norm_range / box_size)
        else:
            # UNIT SPHERE NORMALIZATION:
            if self.zero_mean:
                centroid = torch.mean(coords, axis=0)
                coords = coords - centroid
            if self.scale_factor is not None:
                max_distance = self.scale_factor
            else:
                # max_distance = torch.max(abs(coords_normalized)) / self.norm_range  ## INCORRECT, DOES NOT CONSIDER RADIAL DISTANCE
                max_distance = torch.max(torch.linalg.norm(coords, dim=1)) / self.norm_range
            coords_normalized = coords / max_distance        
        return coords_normalized    
    