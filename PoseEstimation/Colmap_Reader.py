import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from trimesh.scene.cameras import Camera
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

class ColmapReader:
    def __init__(self, path_to_database):
        self.path_to_database = path_to_database

    def get_sparse_model(self):
        try: 
            self.sparse_model_path = os.path.join(self.path_to_database , 'sparse')
        except:
            raise FileNotFoundError('There is no sparse reconstruction in the database, please check if you completely run the colmap code')
    
    def get_dense_model(self):
        try:
            self.dense_model_path = os.path.join(self.path_to_database , 'dense')
        except:
            raise FileNotFoundError('There is no dense reconstruction in the database, please check if you completely run the colmap code')

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