# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# For information on dataset see: https://github.com/mikacuy/pointnetvlad
# Warsaw University of Technology

import numpy as np
import torchvision.transforms as transforms

from datasets.augmentation import JitterPoints, RemoveRandomPoints, RandomTranslation, RemoveRandomBlock, RandomRotation, RandomFlip, Normalize
from datasets.base_datasets import TrainingDataset
from datasets.CSWildPlaces.CSWildPlaces_raw import CSWildPlacesPointCloudLoader


class CSWildPlacesTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = CSWildPlacesPointCloudLoader()


class TrainTransform:
    # Augmentations specific for CSWildPlaces dataset
    def __init__(self, aug_mode, normalize_points=False, scale_factor=None,
                 unit_sphere_norm=False, zero_mean=True, random_rot_theta: float = 5.0):
        self.aug_mode = aug_mode
        self.normalize_points = normalize_points
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        if scale_factor is not None:
            self.normalize_points = True
            self.scale_factor = scale_factor
        self.transform = None
        t = []
        if self.normalize_points:
            t.append(Normalize(scale_factor=self.scale_factor,
                               unit_sphere_norm=self.unit_sphere_norm,
                               zero_mean=self.zero_mean))
        if self.aug_mode == 1:
            # Augmentations without random rotation around z-axis (values assume [-1, 1] range)
            t.extend([JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                      RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)])
        elif self.aug_mode == 2:
            # Augmentations with random rotation around z-axis 
            t.extend([JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                      RandomRotation(max_theta=random_rot_theta, axis=np.array([0, 0, 1])),
                      RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)])
        elif self.aug_mode == 0:    # No augmentations
            pass
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        if len(t) == 0:
            return None
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class ValTransform:
    # Augmentations specific for CSWildPlaces dataset
    def __init__(self, normalize_points=False, scale_factor=None,
                 unit_sphere_norm=False, zero_mean=True):
        self.normalize_points = normalize_points
        self.scale_factor = None
        self.unit_sphere_norm = unit_sphere_norm
        self.zero_mean = zero_mean
        if scale_factor is not None:
            self.normalize_points = True
            self.scale_factor = scale_factor
        t = None
        if self.normalize_points:
            t = Normalize(scale_factor=self.scale_factor,
                          unit_sphere_norm=self.unit_sphere_norm,
                          zero_mean=self.zero_mean)
        self.transform = t

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e