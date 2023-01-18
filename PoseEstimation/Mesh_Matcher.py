import numpy as np
import os
import trimesh
import open3d as o3d
from sys import platform
from glob import glob
import copy
from scipy.spatial.transform import Rotation as R
import json
from evaluate import *


# The file is substituted  with mesh_alignment.py
'''The script is for the mesh alignment task. 
   The pipeline is mainly based on the built-in functions of open3d'''

class MeshAlignment:
    def __init__(self, source, target, num_points, scale_factor, voxel_size = 0.05, icp_method = "standard", scaling = False, normalization = True, filtering = False):
        self.source_mesh = source
        self.target_mesh = target
        self.voxel_size = voxel_size

        # If scaling has to be estimated as well
        self.scaling = scaling

        # If point clouds has to be normalized
        self.normalization = normalization

        # The number of points sampled from the mesh
        self.num_points = num_points

        # If point cloud is need to filtered
        self.filtering = filtering

        # Which registration method is used: standard, multi-scale
        self.icp_method = icp_method

        self.scale_factor = scale_factor



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
    def draw_registration_result(source, target, transformation, colored = True, inverse = False):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        if colored:
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])

        if inverse and not colored:
            colors = source_temp.vertex_colors
            new_colors = np.flip(np.array(colors), 1)

            source_temp.vertex_colors = o3d.utility.Vector3dVector(new_colors)

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

    @staticmethod
    def filter_points(pcd):
        print("Statistical oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30,
                                                            std_ratio=0.2)
        return cl

    '''This function is actually not used'''
    @staticmethod 
    def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    @staticmethod
    def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling = False),3, \
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    @staticmethod
    def refine_registration_multiscale(source, target, result_ransac):
        voxel_radius = [0.02, 0.01, 0.005]
        max_iter = [2000, 1000, 200]
        current_transformation = result_ransac.transformation
        print("3. Colored point cloud registration")
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print([iter, radius, scale])

            print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            print("3-2. Estimate normal.")  
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            print("3-3. Applying colored point cloud registration")
            # result_icp = o3d.pipelines.registration.registration_colored_icp(
            #     source_down, target_down, radius, current_transformation,
            #     o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            #     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
            #                                                     relative_rmse=1e-6,
            #                                                     max_iteration=iter))
            
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-9,
                                                                relative_rmse=1e-9,
                                                                max_iteration=iter))
                                
            current_transformation = result_icp.transformation
        return result_icp

    @staticmethod
    def refine_registration(source, target, result_ransac, voxel_size):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        # result = o3d.pipelines.registration.registration_icp(
        #     source, target, distance_threshold, result_ransac.transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # Built-in function for computing normals for pcd
        radius_normal = voxel_size * 2
        source.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid( radius = radius_normal, max_nn = 30))
        target.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid( radius = radius_normal, max_nn = 30))

        ## Point-to-Plane 
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        ## Colored ICP
        # result = o3d.pipelines.registration.registration_icp(
        #     source, target, distance_threshold, result_ransac.transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint())

        ## Generalized ICP
        # result = o3d.pipelines.registration.registration_generalized_icp(
        #     source, target, distance_threshold, result_ransac.transformation,
        #     o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())

        ## Point-to-Point
        # result = o3d.pipelines.registration.registration_icp(
        #     source, target, distance_threshold, result_ransac.transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
            
        return result

    def normalize_pc(self):
        points_target = np.asarray(self.target.points)
        normalized_points_target, centroid_gt, normalize_factor_target = self.normalize_points(points_target)
        self.target.points = o3d.utility.Vector3dVector(normalized_points_target)

        points_reconstruction = np.asarray(self.source.points)
        normalized_points_reconstruction, centroid_rc, normalize_factor_reconstruction = self.normalize_points(points_reconstruction)
        self.source.points = o3d.utility.Vector3dVector(normalized_points_reconstruction)

        self.additional_factor = normalize_factor_reconstruction/ normalize_factor_target


        # Parameters for recorvering the normalization and scaling effect for evalutation of translation vector
        self.backward_factor = normalize_factor_target
        self.centroid_gt = centroid_gt
        self.centroid_rc = centroid_rc


    def prepare_dataset(self):
        print(":: Sample point clouds from two meshes")
        self.source = self.sample_pc(self.source_mesh, self.num_points)
        self.target = self.sample_pc(self.target_mesh, self.num_points)

        if self.filtering:
            print("Statistical oulier removal")
            self.source = self.filter_points(self.source)
            self.target = self.filter_points(self.target)
        
        if self.normalization:
            print(":: Normalize point clouds")
            self.normalize_pc()
            scale_factor = self.scale_factor * self.additional_factor
        
        else:
            scale_factor = self.scale_factor
        self.source.scale(scale_factor, center=np.zeros(3))
        #self.source.scale(scale_factor, center=self.source.get_center())


        self.source_down, self.source_fpfh = self.preprocess_point_cloud(self.source, self.voxel_size)
        self.target_down, self.target_fpfh = self.preprocess_point_cloud(self.target, self.voxel_size)

    def register(self):
        self.prepare_dataset()
        
        self.result_ransac = self.execute_global_registration(self.source_down, self.target_down,
                                                self.source_fpfh, self.target_fpfh,
                                                self.voxel_size)

        if self.icp_method == "standard":
            self.result_icp = self.refine_registration(self.source, self.target, self.result_ransac, self.voxel_size)
        elif self.icp_method == "multiscale":
            self.result_icp = self.refine_registration_multiscale(self.source, self.target, self.result_ransac)
        else:
            raise NotImplementedError("Please check the name is given correctly")


if __name__ == "__main__":
    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
    # linux
        path_reconstruction = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/output/0/meshed-poisson.ply")
        path_gt = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/gt/*.ply")
        path_gt_transformation = "/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + '/output/gt_transformation.json'

    elif platform == "win32":
    # Windows...
        path_gt = glob('D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/GT/*.ply')
        path_reconstruction = glob('D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/Output/meshed-poisson.ply')

        path_gt_transformation = 'D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/Output/gt_transformation.json'


    mesh_reconstruction = o3d.io.read_triangle_mesh(path_reconstruction[-1])
    mesh_gt = o3d.io.read_triangle_mesh(path_gt[-1])
    num_points = 500000

    # Load the ground truth transformation
    with open(path_gt_transformation, 'r') as fp:
        gt_transformation = json.load(fp)



    Aligner = MeshAlignment(mesh_reconstruction, mesh_gt, num_points, scale_factor = gt_transformation['scale'], voxel_size=0.05, icp_method="standard")

    Aligner.register()
    #Aligner.draw_registration_result(Aligner.source_down, Aligner.target_down, Aligner.result_ransac.transformation)
    print(Aligner.result_ransac)
    print(Aligner.result_icp)
    


    # We need to recorver the translation because the point cloud we used in aligner is normalized and prescaled.
    # Rotation is not affected by those factors
    delta_tran = Aligner.centroid_gt - Aligner.result_icp.transformation[:3, :3] @ (Aligner.centroid_rc * Aligner.scale_factor)
    tran_new = Aligner.result_icp.transformation[:3, 3].reshape(3, 1) * Aligner.backward_factor + delta_tran.reshape(3, 1)
    
    rotError = rotation_error(np.array(gt_transformation['rotation']), Aligner.result_icp.transformation[:3, :3])
    tranError = position_error(np.array(gt_transformation['translation']).reshape(3, 1), tran_new) 

    print("Rotation Error is {} degree and Translation Error is {} meters".format(rotError, tranError))

    
    est_new = np.concatenate(
                    [np.concatenate([Aligner.result_icp.transformation[:3, :3], tran_new], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    mesh_reconstruction.scale(gt_transformation['scale'], center = np.zeros(3))
    Aligner.draw_registration_result(mesh_reconstruction, mesh_gt, est_new, colored=False, inverse=True)

    
