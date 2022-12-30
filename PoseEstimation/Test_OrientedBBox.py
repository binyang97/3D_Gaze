import numpy as np
import os
import trimesh
import open3d as o3d
from sys import platform
from glob import glob
import copy 

def sample_pc(mesh, num_points, method = "Uniform"):
    if method == "Uniform":
        pcd_sample = mesh.sample_points_uniformly(number_of_points=num_points)
    elif method == "PoissonDisk":
        pcd_sample = mesh.sample_points_poisson_disk(number_of_points=int(num_points/5), init_factor=5)
    
    else:
        raise NameError('The method used for sampling point cloud is not available, please check it')

    return pcd_sample

def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return points, centroid, furthest_distance

if __name__ == "__main__"  : 
    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
    # linux
        path_reconstruction = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/output/*/fused.ply")
        path_gt = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/gt/*.ply")

    elif platform == "win32":
    # Windows...
        path_gt = glob("D:/Documents/Semester_Project/Colmap_Test/GT/*.ply")
        path_reconstruction = glob("D:/Documents/Semester_Project/Colmap_Test/Output/fused.ply")


    pcd_reconstruction = o3d.io.read_point_cloud(path_reconstruction[-1])
    #mesh_reconstruction = o3d.io.read_triangle_mesh(path_reconstruction[-1])
    mesh_gt = o3d.io.read_triangle_mesh(path_gt[-1])

    #number_of_points = len(pcd_reconstruction.points)
    number_of_points = 500000
    pcd_sample = sample_pc(mesh_gt, number_of_points)


    points_sample = np.asarray(pcd_sample.points)
    normalized_points_sample, _, _ = normalize_pc(points_sample)
    pcd_sample.points = o3d.utility.Vector3dVector(normalized_points_sample)

    points_reconstruction = np.asarray(pcd_reconstruction.points)
    normalized_points_reconstruction, _, _ = normalize_pc(points_reconstruction)
    pcd_reconstruction.points = o3d.utility.Vector3dVector(normalized_points_reconstruction)


    cl, ind = pcd_reconstruction.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=0.2)
    pcd_reconstruction = copy.deepcopy(cl)

    cl, ind = pcd_sample.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=0.2)
    pcd_sample = copy.deepcopy(cl)


    BBox1 = pcd_reconstruction.get_oriented_bounding_box()
    BBox1.color = [1, 0, 0]
    BBox2 = pcd_sample.get_oriented_bounding_box()
    BBox2.color = [0, 1, 0]

    print(BBox1.R, BBox2.R)


    print(BBox2.get_min_bound(), BBox1.get_min_bound())
    print(BBox2.get_max_bound(), BBox1.get_max_bound())
    o3d.visualization.draw_geometries([BBox1, BBox2, pcd_sample, pcd_reconstruction])


    # extents_bbox2 = BBox2.get_max_bound() - BBox2.get_min_bound()
    # extents_bbox1 = BBox1.get_max_bound() - BBox1.get_min_bound()

    # print(extents_bbox1, extents_bbox2)

    # print(BBox1.volume(), BBox2.volume())


    