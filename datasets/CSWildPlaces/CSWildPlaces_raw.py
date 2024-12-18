import numpy as np
import os
import open3d as o3d

from datasets.base_datasets import PointCloudLoader


class CSWildPlacesPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        assert os.path.splitext(file_path)[-1] == ".pcd"
        pc_o3d = o3d.io.read_point_cloud(file_path)
        pc = np.asarray(pc_o3d.points, dtype=np.float64)
        pc = np.float32(pc)
        
        return pc
