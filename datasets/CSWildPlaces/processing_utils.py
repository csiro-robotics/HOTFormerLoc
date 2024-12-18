import time
from multiprocessing import Pool
from typing import Tuple, Optional

import numpy as np
import open3d as o3d 
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from numba import jit, njit
import numba
import CSF

RANDOM_SEED = 42
CSF_RIGIDNESS = 2
CSF_THRESHOLD = 0.5     # m from cloth to classify as ground
CSF_RESOLUTION = 1.0      # m
CSF_BSLOOPSMOOTH = True
VOXEL_STEP = 0.01       # for pnvlad downsampling
CLOUD_SAVE_DIR = 'clouds/'
POSES_FILENAME = "poses.csv"

def quaternion_to_rot(quat_transform):
    # qx, qy, qz, qw format
    xyz = quat_transform[:3]
    quaternion = quat_transform[3:]

    r = R.from_quat(quaternion)
    rot_matrix = r.as_matrix()
    trans = np.concatenate([rot_matrix, xyz.reshape(3,1)], axis = 1)
    trans = np.concatenate([trans, np.array([[0.,0.,0.,1.]])], axis = 0)
    return trans

def rot_to_quaternion(rot_transform):
    r = R.from_matrix(rot_transform[:3,:3])
    quat = r.as_quat()
    xyz = rot_transform[:3,3]
    return xyz, quat

def make_o3d_pcl(points: np.ndarray):
    assert len(points) > 0, 'Cannot make point cloud from 0 points'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def viz_o3d_pcl(cloud):
    o3d.visualization.draw_geometries([cloud])    
    return None

def display_cloud_idx_pts(cloud, idx):
    """
    Highlight points in a cloud by index (could be for outlier removal, ground 
    filtering, etc)
    """
    inlier_cloud = cloud.select_by_index(idx, invert=True)
    outlier_cloud = cloud.select_by_index(idx)

    print("Showing index pts (brown) and inliers (colormap): ")
    outlier_cloud.paint_uniform_color([135/255, 0/255, 173/255])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return None

def remove_ground_CSF(pts: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Remove ground points using the Cloth Simulation Filter method.
    """
    csf = CSF.CSF()
    csf.params.bSloopSmooth = CSF_BSLOOPSMOOTH
    csf.params.cloth_resolution = CSF_RESOLUTION
    csf.params.rigidness = CSF_RIGIDNESS
    csf.params.threshold = CSF_THRESHOLD
    
    csf.setPointCloud(pts)
    ground = CSF.VecInt()       # index of ground points
    non_ground = CSF.VecInt()   # index of non-ground points
    csf.do_filtering(ground, non_ground, exportCloth=False)

    if len(np.array(non_ground)) > 0:
        filtered_pts = pts[np.array(non_ground)] # extract non-ground points
    else:
        filtered_pts = np.array([])  # handle case where all pts are ground
    
    if debug:
        cloud = make_o3d_pcl(pts)
        display_cloud_idx_pts(cloud, np.array(ground))
    
    return filtered_pts

def random_down_sample(
    points: np.ndarray,
    downsample_number: int,
    random_seed = RANDOM_SEED
) -> np.ndarray:
    """
    Randomly downsample point cloud to desired number of points.
    """
    rng = np.random.default_rng(seed=random_seed)
    points_downsampled = rng.choice(points, downsample_number)
    return points_downsampled

def pnvlad_down_sample(
    points: np.ndarray,
    downsample_number: int,
    random_seed = RANDOM_SEED
) -> np.ndarray:
    """
    Iteratively voxel downsample point cloud to desired number of points, 
    following the method used by PointNetVLAD authors.
    """
    rng = np.random.default_rng(seed=random_seed)
    voxel_size = 3.001  # initialise at very large voxels
    cloud = make_o3d_pcl(points)
    
    # Find suitable voxel size iteratively
    cloud_downsampled = cloud.voxel_down_sample(voxel_size)
    while len(cloud_downsampled.points) < downsample_number:
        voxel_size -= VOXEL_STEP
        if voxel_size <= 0:
            raise AssertionError(f"Cloud size {len(cloud_downsampled.points)} is somehow smaller than {downsample_number} with 1cm voxels")
            """
            cloud_downsampled = cloud
            break
            """
        cloud_downsampled = cloud.voxel_down_sample(voxel_size)
    
    while len(cloud_downsampled.points) > downsample_number:
        voxel_size += VOXEL_STEP / 5   # fine-grained steps for refinement
        cloud_downsampled = cloud.voxel_down_sample(voxel_size)
    
    # print(f"Voxel size: {voxel_size:.3f}") # NOTE: DEBUG
    # Add additional random points to pad to desired NUM_POINTS
    num_extra_points = downsample_number - len(cloud_downsampled.points)
    # print(f"Num extra points: {num_extra_points:.3f}") # NOTE: DEBUG
    points = np.asarray(cloud.points)
    points_downsampled = np.asarray(cloud_downsampled.points)
    rand_points = rng.choice(points, size=num_extra_points)
    points_downsampled = np.concatenate((points_downsampled, rand_points))
    
    return points_downsampled

def voxel_down_sample(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """
    Voxel downsample point cloud to desired voxel size.
    """
    cloud = make_o3d_pcl(points)
    cloud_downsampled = cloud.voxel_down_sample(voxel_size)
    points_downsampled = np.asarray(cloud_downsampled.points)
    return points_downsampled

def remove_outliers(
    points: np.ndarray,
    points_timestamps: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform statistical outlier removal on input pointcloud, and return filtered
    points (and optionally filtered timestamps.)
    """
    cloud = make_o3d_pcl(points)
    cloud_filtered, indices = cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=3.0)
    points_filtered = np.asarray(cloud_filtered.points)
    if points_timestamps is not None:
        points_timestamps = points_timestamps[indices]
        assert len(points_filtered) == len(points_timestamps)
    
    return points_filtered, points_timestamps

def normalise_pcl(
    points_downsampled: np.ndarray,
    points: np.ndarray,
    downsample_number: int,
    random_seed = RANDOM_SEED
) -> np.ndarray:
    """
    Method for normalising submaps to [-1, 1] range, proposed in PointNetVLAD.
    """
    ###### original method from paper
    rng = np.random.default_rng(seed=random_seed)
    centroid = np.mean(points_downsampled, 0)    
    # Make spread s=0.5/d
    sum = 0
    for i in range(len(points_downsampled)):
        sum += np.sqrt(
            (points_downsampled[i,0]-centroid[0])**2
            + (points_downsampled[i,1]-centroid[1])**2
            + (points_downsampled[i,2]-centroid[2])**2
        )
    d = sum / len(points_downsampled)
    s = 0.5 / d

    T = np.array([[s, 0, 0, -s*(centroid[0])],
                 [0, s, 0, -s*(centroid[1])],
                 [0, 0, s, -s*(centroid[2])],
                 [0, 0, 0, 1]])
    cloud_scaled = make_o3d_pcl(points_downsampled)
    cloud_scaled.transform(T)
    pts_scaled = np.asarray(cloud_scaled.points)
    
    # Enforce to be in [-1, 1]
    pts_final = pts_scaled[np.all(np.abs(pts_scaled) <= 1, axis=1)]
    
    # Add points as necessary
    if downsample_number is not None:
        num_extra_points = downsample_number - len(pts_final)
        pts_final = np.copy(pts_final)
        points_added = 0
        while len(pts_final) < downsample_number:
            rand_points = rng.choice(points, size=(num_extra_points - points_added))
            cloud_rand = make_o3d_pcl(rand_points)
            cloud_rand.transform(T)
            rand_points = np.asarray(cloud_rand.points)
            # keep valid points, otherwise loop again
            rand_points = rand_points[np.all(np.abs(rand_points) <= 1, axis=1)]
            points_added += len(rand_points)
            pts_final = np.concatenate((pts_final, rand_points))
        
        assert(len(pts_final) == downsample_number), \
            f"normalisation error, size {len(pts_final)}"
    ######
    
    assert(pts_final.min() >= -1 and pts_final.max() <= 1), \
        "normalisation error"
    
    return pts_final

@jit(nopython=True)
def points_in_polygon(polygon, point):
    """
    Function from:
    https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py,
    found in:
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    """
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/below/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return intersections & 1

@njit(parallel=True)
def points_in_polygon_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = points_in_polygon(polygon,points[i])
    return D        

def multiprocessing_func(function, inputs, num_workers):
    tic = time.time()
    with Pool(num_workers) as p:
        results = list(tqdm(p.imap(function, inputs), total = len(inputs)))
        p.close()
        p.join()
    toc = time.time()
    print(f'Runtime: {toc - tic:.2f}s')
    return results
