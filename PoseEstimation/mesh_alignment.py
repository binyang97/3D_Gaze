import numpy as np
import os
import trimesh
import open3d as o3d
from sys import platform
from glob import glob


def sample_pc(mesh, num_points, method = "Uniform"):
    if method == "Uniform":
        pcd_sample = mesh.sample_points_uniformly(number_of_points=num_points)
    elif method == "PoissonDisk":
        pcd_sample = mesh_gt.sample_points_poisson_disk(number_of_points=int(num_points/5), init_factor=5)
    
    else:
        raise NameError('The method used for sampling point cloud is not available, please check it')

    return pcd_sample



if __name__ == "__main__":
    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
    # linux
        path_reconstruction = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/output/*/*.ply")
        path_gt = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/gt/*.ply")

    elif platform == "win32":
    # Windows...
        txt_filepath = 'D:/Documents/Semester_Project/Colmap_Test/Output/images.txt'
    
    pcd_reconstruction = o3d.io.read_point_cloud(path_reconstruction[-1])
    mesh_gt = o3d.io.read_triangle_mesh(path_gt[-1])


    # Visualization --> The scale of two 3D models are different
    # pcd.paint_uniform_color([1, 0, 0])

    # #o3d.visualization.draw_geometries([mesh_gt])
    # o3d.visualization.draw_geometries([pcd, mesh_gt])

    # Sample Point cloud from mesh

    number_of_points = 100000
    pcd_sample = sample_pc(mesh_gt, number_of_points)
    o3d.visualization.draw_geometries([pcd_sample])

    
    reg_p2p = o3d.pipelines.registration.registration_icp( \
        source, target, threshold, trans_init, \
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scale = True))


