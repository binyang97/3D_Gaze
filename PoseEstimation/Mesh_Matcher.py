import numpy as np
import os
import trimesh
import open3d as o3d
from sys import platform
from glob import glob
import copy
from scipy.spatial.transform import Rotation as R


class MeshAlignment:
    def __init__(self, source, target, num_points, voxel_size = 0.05, scaling = False, normalization = True):
        self.source_mesh = source
        self.target_mesh = target
        self.voxel_size = voxel_size

        # If scaling has to be estimated as well
        self.scaling = scaling

        # If point clouds has to be normalized
        self.normalization = normalization

        # The number of points sampled from the mesh
        self.num_points = num_points


    @staticmethod
    def sample_pc(mesh, num_points, method = 'Uniform'):
        if method == "Uniform":
            pcd_sample = mesh.sample_points_uniformly(number_of_points=num_points)
        elif method == "PoissonDisk":
            pcd_sample = mesh.sample_points_poisson_disk(number_of_points=int(num_points/5), init_factor=5)
        
        else:
            raise NameError('The method used for sampling point cloud is not available, please check it')

        return pcd_sample

    @staticmethod
    def normalize_points(points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

        return points, centroid, furthest_distance

    @staticmethod
    def draw_registration_result(source, target, transformation, colored = True):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        if colored:
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    @staticmethod
    def save_registration_result(source, target, transformation, filename):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        output = source_temp + target_temp
        o3d.io.write_point_cloud(filename, output)
        print("point cloud is saved")

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        print(len(pcd_down.points))

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def normalize_pc(self):
        points_target = np.asarray(self.target.points)
        normalized_points_target, _, normalize_factor_target = self.normalize_points(points_target)
        self.target.points = o3d.utility.Vector3dVector(normalized_points_target)

        points_reconstruction = np.asarray(self.source.points)
        normalized_points_reconstruction, _, normalize_factor_reconstruction = self.normalize_points(points_reconstruction)
        self.source.points = o3d.utility.Vector3dVector(normalized_points_reconstruction)

        self.additional_factor = normalize_factor_reconstruction/ normalize_factor_target

    def prepare_dataset(self):
        print(":: Sample point clouds from two meshes")
        self.source = self.sample_pc(self.source_mesh)
        self.target = self.sample_pc(self.target_mesh)
        if self.normalization:
            print(":: Normalize point clouds")
            self.normalize_pc()

        self.source_down, self.source_fpfh = self.preprocess_point_cloud(self.source, self.voxel_size)
        self.target_down, self.target_fpfh = self.preprocess_point_cloud(self.target, self.voxel_size)

    def execute_global_registration(self):
        distance_threshold = self.voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % self.voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.source_down, self.target_down, self.source_fpfh, self.target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling = self.scaling),3, \
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result