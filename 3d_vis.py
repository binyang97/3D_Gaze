import numpy as np
import trimesh


point_cloud = np.load('/home/biyang/Documents/3D_Gaze/dataset/point_cloud_no_name.npz', allow_pickle=True)

points = point_cloud['points']



#test_points = point_cloud[list(point_cloud.keys())[0]][1][0]
#points = np.array(points)
#print(points.shape)

# Subsampling of the point clouds
print(len(points))
factor=80
decimated_points_random = points[::factor]
print(len(decimated_points_random))
pc = trimesh.PointCloud(decimated_points_random)
scene = trimesh.Scene(pc)

scene.show(line_settings={'point_size':0.5})