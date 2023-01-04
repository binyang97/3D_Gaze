import numpy as np
import os
import trimesh
import open3d as o3d
from sys import platform
from glob import glob
import copy
from scipy.spatial.transform import Rotation as R

def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return points, centroid, furthest_distance

def sample_pc(mesh, num_points, method = "Uniform"):
    if method == "Uniform":
        pcd_sample = mesh.sample_points_uniformly(number_of_points=num_points)
    elif method == "PoissonDisk":
        pcd_sample = mesh.sample_points_poisson_disk(number_of_points=int(num_points/5), init_factor=5)
    
    else:
        raise NameError('The method used for sampling point cloud is not available, please check it')

    return pcd_sample


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


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


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4559,
    #                                   front=[0.6452, -0.3036, -0.7011],
    #                                   lookat=[1.9892, 2.0208, 1.8945],
    #                                   up=[-0.2779, -0.9482, 0.1556])


def save_registration_result(source, target, transformation, filename):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    output = source_temp + target_temp
    o3d.io.write_point_cloud(filename, output)
    print("point cloud is saved")

def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    #demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling = True),3, \
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

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

def refine_registration_multiscale(source, target, result_ransac):
    voxel_radius = [0.02, 0.01, 0.005]
    max_iter = [200, 100, 20]
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

if __name__ == "__main__":
    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
    # linux
        path_reconstruction = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/output/*/meshed-poisson.ply")
        path_gt = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/gt/*.ply")

    elif platform == "win32":
    # Windows...
        path_gt = glob('D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/GT/*.ply')
        path_reconstruction = glob('D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/Output/meshed-poisson.ply')

    ALIGNMENT = True
    FILTERING = False
    FAST = False
    SAVE_PCD = False
    SAVE_REGISTRATION = False
    NORMALIZATION = True
    ICP_METHOD = "icp_standard" # or "icp_standard"
    SCALING = True
    
    #pcd_reconstruction = o3d.io.read_point_cloud(path_reconstruction[-1])
    mesh_reconstruction = o3d.io.read_triangle_mesh(path_reconstruction[-1])
    mesh_gt = o3d.io.read_triangle_mesh(path_gt[-1])


    #number_of_points = len(pcd_reconstruction.points)
    number_of_points = 500000
    pcd_sample = sample_pc(mesh_gt, number_of_points)
    pcd_reconstruction = sample_pc(mesh_reconstruction, number_of_points)

    ## Test for gt scale
    # r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    # T_random = np.array([[1], [1.5], [2.5]])
    # trans_random = np.concatenate(
    #                     [np.concatenate([r.as_matrix(), T_random], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    # pcd_reconstruction = sample_pc(mesh_gt, 200000)
    
    # pcd_reconstruction.transform(trans_random)

    # o3d.visualization.draw_geometries([pcd_reconstruction, pcd_sample])

    if SAVE_PCD:

        o3d.io.write_point_cloud("/home/biyang/Documents/3D_Gaze/Colmap/40777060/pointcloud/gt_100000.ply", pcd_sample)
        o3d.io.write_point_cloud("/home/biyang/Documents/3D_Gaze/Colmap/40777060/pointcloud/reconstruction_100000.ply", pcd_reconstruction)


    if NORMALIZATION:
        points_sample = np.asarray(pcd_sample.points)
        normalized_points_sample, _, normalize_factor_sample = normalize_pc(points_sample)
        pcd_sample.points = o3d.utility.Vector3dVector(normalized_points_sample)

        points_reconstruction = np.asarray(pcd_reconstruction.points)
        normalized_points_reconstruction, _, normalize_factor_reconstruction = normalize_pc(points_reconstruction)
        pcd_reconstruction.points = o3d.utility.Vector3dVector(normalized_points_reconstruction)

        #o3d.visualization.draw_geometries([pcd_reconstruction, pcd_sample])
        #draw_registration_result(pcd_sample, pcd_reconstruction, np.eye(4))

    if SCALING:
    ## Prescaling is useful, but it is not enough only with normalization
        additional_scale_factor = normalize_factor_reconstruction / normalize_factor_sample
        pcd_reconstruction_scaled = copy.deepcopy(pcd_reconstruction)
        pcd_reconstruction_scaled.scale(0.4508734833211593 * additional_scale_factor, center=pcd_reconstruction_scaled.get_center())
        pcd_reconstruction = copy.deepcopy(pcd_reconstruction_scaled)

    # draw_registration_result(pcd_sample, pcd_reconstruction, np.eye(4))

    if FILTERING:
        print("Statistical oulier removal")
        cl, ind = pcd_reconstruction.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=0.1)
        #display_inlier_outlier(pcd_reconstruction, ind)
        print(len(pcd_reconstruction.points))
        print(len(cl.points))
        pcd_reconstruction = cl

        cl_gt, ind = pcd_sample.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=0.1)
        print(len(pcd_sample.points))
        print(len(cl_gt.points))
        pcd_sample = cl_gt
        

    # print(np.array(pcd_reconstruction.points).max(axis = 0))     
    # print(np.array(pcd_sample.points).max(axis = 0))  
        #o3d.visualization.draw_geometries([cl])

    # Sample Point cloud from mesh

    
    # Downsample the pcd for global registration

    if ALIGNMENT:

        voxel_size = 0.05
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcd_reconstruction, pcd_sample, voxel_size)

        #o3d.visualization.draw_geometries([source_down, target_down])

        if FAST:
            result_ransac = execute_fast_global_registration(source_down, target_down,
                                               source_fpfh, target_fpfh,
                                               voxel_size)
        else:
            result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        print(result_ransac)
        draw_registration_result(source_down, target_down, result_ransac.transformation)

        if SAVE_REGISTRATION:
            save_registration_result(source_down, target_down, result_ransac.transformation, "/home/biyang/Documents/results/global_registration.ply")
        if ICP_METHOD == "icp_standard":
            result_icp = refine_registration(source, target, result_ransac, voxel_size)
        elif ICP_METHOD == "icp_multiscale":
            result_icp = refine_registration_multiscale(source, target, result_ransac)
        else:
            raise NotImplementedError
        print(result_icp)
        if SAVE_REGISTRATION:
            save_registration_result(source, target, result_icp.transformation, "/home/biyang/Documents/results/local_refinement.ply")
        draw_registration_result(source, target, result_icp.transformation)


