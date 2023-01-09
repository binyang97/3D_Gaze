import numpy as np
import os
import trimesh
import open3d as o3d
from sys import platform
from glob import glob
import copy
from scipy.spatial.transform import Rotation as R
from sys import platform
import pandas as pd
import cv2
import json


# The script is mainly used for preprocessing the raw video data recorded by Pupil Invisible
'''1. Extract Image Sequence with respect to timestamp of recorded gaze info
   2. Rewrite Gaze data with pre-defined step size
   3. Save the IMU data '''
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class DataPreprocessor:
    def __init__(self, path_to_raw_data, output_path):
        self.rawdata_path = path_to_raw_data
        self.output_path = output_path
        self.__export_info__()
        self.__get_timestamps__()
        self.__get_gaze__()

    def __export_info__(self):
        export_info_path = os.path.join(datapath, 'export_info.csv')
        if not os.path.exists(export_info_path):
            raise FileNotFoundError('Cannot find export_info.csv, please check the given filepath is correct or the file is correctly exported from Pupil Player')

        info_df = pd.read_csv(export_info_path)

        time_range = info_df.loc[info_df.key == 'Absolute Time Range','value'].item()

        self.starttime = float(time_range.split(' - ')[0])
        self.endtime = float(time_range.split(' - ')[1])

    
        frame_index_range = info_df.loc[info_df.key == 'Frame Index Range:','value'].item()
        self.startid = float(frame_index_range.split(' - ')[0])
        self.endid = float(frame_index_range.split(' - ')[0])

    def __get_timestamps__ (self):
        timestamps_path = os.path.join(datapath, 'world_timestamps.csv')
        if not os.path.exists(timestamps_path):
            raise FileNotFoundError('Cannot find world_timestamps.csv, please check the given filepath is correct or the file is correctly exported from Pupil Player')

        ts_df = pd.read_csv(timestamps_path)
        self.world_ts = ts_df.iloc[:,0].values
        self.world_pts = ts_df.iloc[:, 1].values

    # def __get_imu__(self):
    #     gaze_path = os.path.join(datapath, 'imu_data.csv')
    #     if not os.path.exists(gaze_path):
    #         raise FileNotFoundError('Cannot find imu_data.csv, \
    #                                 please check the given filepath is correct or the file is correctly exported from Pupil Player')   

    def __get_gaze__(self):
        gaze_path = os.path.join(datapath, 'gaze_positions.csv')
        if not os.path.exists(gaze_path):
            raise FileNotFoundError('Cannot find gaze_positions.csv, please check the given filepath is correct or the file is correctly exported from Pupil Player')

        gaze_df = pd.read_csv(gaze_path)
        self.gaze_df = gaze_df.dropna(axis=1, how = "all")
        #self.gaze_ts = self.gaze_df.iloc[:,0].values


    def extract_frames(self):
        video_path = os.path.join(datapath, 'world.mp4')
        save_image_path = os.path.join(self.output_path, 'images')
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError('Cannot find world.mp4, please check the given filepath is correct or the file is correctly exported from Pupil Player')
        
        vid = cv2.VideoCapture(video_path)

        ret, img = vid.read()
        count = 0
        while ret:
            cv2.imwrite(os.path.join(save_image_path, "{:05n}.jpg".format(count)), img)
            ret, img = vid.read()
            count += 1
        
        print('Frames are successfully extracted')
            # print(vid.get(cv2.CAP_PROP_POS_MSEC))
            # print(vid.get(cv2.CAP_PROP_POS_FRAMES))
            # print(vid.get(cv2.CAP_PROP_FPS))

    def extract_gaze(self):

        self.gaze_dict = {}
        for i, timestamp in enumerate(self.world_ts):
            filtered_df = self.gaze_df[self.gaze_df['world_index'] == (i + self.startid)]
            filtered_gaze_ts = np.array(filtered_df.iloc[:,0].values)
            
            target_idx = find_nearest(filtered_gaze_ts, timestamp)
            gaze_info = filtered_df.iloc[target_idx]

            assert gaze_info['world_index']-self.startid == i

            
            #gaze_info.dropna().reset_index(drop=True)

            self.gaze_dict["{:05n}".format(i)] = gaze_info.to_dict()

        gaze_output_path = os.path.join(self.output_path, 'gaze.json')
        with open(gaze_output_path, 'w') as fp:
            json.dump(self.gaze_dict, fp)

        print('Gaze data is succesfully exported')
            






        




if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":  
    # linux
        datapath = r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/raw_data/2021-05-20-15-19-08/exports/000"
        output_path = r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/room1"

    elif platform == "win32":
    # Windows...
        datapath = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\raw_data\2021-05-20-15-19-08\exports\000"
        output_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible"

    


    # ts_df = pd.read_csv(os.path.join(datapath, 'world_timestamps.csv'))
    
    # time_range = df.loc[df.key == 'Absolute Time Range','value'].item()
    # starttime = float(time_range.split(' - ')[0])
    # print(starttime)

    
    data = DataPreprocessor(datapath, output_path)
    data.extract_gaze()
    data.extract_frames()
    