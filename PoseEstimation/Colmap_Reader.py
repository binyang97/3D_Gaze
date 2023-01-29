import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from load_ARSceneData import LoadARSceneData
from sys import platform
from evaluate import position_error, rotation_error
from camera_pose_visualizer import SimpleCameraPoseVisualizer
import matplotlib as plt
import random
from glob import glob
import re
import os
from colmap_read_write_model import read_model, detect_model_format
import open3d as o3d


####### Not in used ######################

'''Read the colmap data, give the path to the workspace of colmap reconstruction
'''

class ColmapReader:
    def __init__(self, path_to_database):
        self.path_to_database = path_to_database
        self.sparse_model_path = self.get_sparse_model()
        self.dense_model_path = self.get_dense_model()


    def get_sparse_model(self):
        sparse_model_path = os.path.join(self.path_to_database , 'sparse')
        if not os.path.exists(sparse_model_path):
            raise FileNotFoundError('There is no sparse reconstruction in the database, please check if you completely run the colmap code')
        
        if detect_model_format(sparse_model_path, ".bin") or detect_model_format(sparse_model_path, ".txt"):
            return sparse_model_path
        else:
            subfolder = os.listdir(sparse_model_path)
            assert len(subfolder) == 1, "There exists multiple folders in sparse folder, please check it"
            sparse_model_path = os.path.join(sparse_model_path, subfolder[0])

            if detect_model_format(sparse_model_path, ".bin") or detect_model_format(sparse_model_path, ".txt"):
                return sparse_model_path
            else:
                raise FileNotFoundError('There is no sparse reconstruction in the database, please check if you completely run the colmap code')

    # ToDo: detect the .ply file
    def get_dense_model(self):
        
        dense_model_path = os.path.join(self.path_to_database , 'dense')
        if not os.path.exists(dense_model_path):
            raise FileNotFoundError('There is no dense reconstruction in the database, please check if you completely run the colmap code')

        if os.path.isfile(os.path.join(dense_model_path, "fused.ply")):
            pass
        else:
            subfolder = os.listdir(dense_model_path)
            assert len(subfolder) == 1, "There exists multiple folders in sparse folder, please check it"

            dense_model_path = os.path.join(dense_model_path, subfolder[0])

        return dense_model_path
    
    def read_sparse_model(self):
        cameras, images, points3D = read_model(self.sparse_model_path)

        return cameras, images, points3D

    def read_dense_model(self):
        pcd = o3d.io.read_point_cloud(os.path.join(self.dense_model_path, "fused.ply"))

        if os.path.isfile(os.path.join(self.dense_model_path, "meshed-poisson.ply")):
            mesh = o3d.io.read_triangle_mesh(os.path.join(self.dense_model_path, "meshed-poisson.ply"))
        else:
            print("There is no possion mesh")
            mesh = None

        return pcd, mesh


    @staticmethod
    def rewrite_image_txt(path_to_file):
        image_id = []
        camera_params = [] # QW, QX, QY, QZ, TX, TY, TZ
        points_2D = [] # 2D coordinates of keypoints
        point3D_IDs = [] # corresponding 3D point id

        with open(path_to_file) as f:
            contents = f.readlines()

        for content in contents:
            if content.__contains__('#'):
                pass
            elif content.__contains__('png') or content.__contains__('jpg'): ## This could be other format such as jpg
                str_list = content.split()
                image_id.append(str_list[-1])
                camera_params.append([float(string) for string in str_list[1:-2]])
            else:
                str_list = np.array(content.split())
                str_list = str_list.reshape(-1, 3)
                point_2D = []
                point3D_ID = []
                for ele in str_list:
                    point_2D.append(ele[:2].astype(np.float64))
                    point3D_ID.append(ele[2].astype(np.int64))
                
                points_2D.append(point_2D)
                point3D_IDs.append(point3D_ID)

        points_2D = np.array(points_2D, dtype=object)
        point3D_IDs = np.array(point3D_IDs, dtype=object)
        camera_params = np.array(camera_params)

        return image_id, camera_params, points_2D, point3D_IDs


if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":  
    # linux
        dataset_path = r""

    elif platform == "win32":
    # Windows...
        dataset_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image_100_undistorted_prerecorded\Stereo_Fusion.min_num_pixels=10"

    data = ColmapReader(dataset_path)
    cameras, images, points3D = data.read_sparse_model()
    # pcd_dense, mesh_dense = data.read_dense_model()
    # o3d.visualization.draw_geometries([pcd_dense])

    #print(cameras.items()[0])
    #print(images.items())
    print(points3D[7754])