import os
import shutil
from sys import platform

import json


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    

if __name__=="__main__":

    if platform == "linux" or platform == "linux2":  
    # linux
        rawdata_path = r"/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/2023_02_03_12_36_28-20230203T204041Z-001/2023_02_03_12_36_28"
        output_path= r"/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app"

    elif platform == "win32":
    # Windows...
        rawdata_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\2023_02_03_12_36_28"
        output_path= r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App"

    
    create_folder(output_path)

    try:
        with open(os.path.join(rawdata_path, "info.json")) as f:
            info = json.load(f)
    except:
        raise FileNotFoundError

    folder = os.path.join(output_path, info["title"])
    
    
    create_folder(folder)
    

    depth_folder = os.path.join(folder, "depth")
    conf_folder = os.path.join(folder, "conf")
    frames_folder = os.path.join(folder, "frames")
    pose_folder = os.path.join(folder, "pose")
    data3d_folder = os.path.join(folder, "data3d")

    create_folder(depth_folder)
    create_folder(conf_folder)
    create_folder(frames_folder)
    create_folder(pose_folder)
    create_folder(data3d_folder)

    all_files = set(os.listdir(rawdata_path))
    
    depth_files = set(filter(lambda file: file.__contains__("depth"), all_files))
    conf_files = set(filter(lambda file: file.__contains__("conf"), all_files))
    frames_files = set(filter(lambda file: file.__contains__("frame") and file.__contains__("jpg"), all_files))
    pose_files = set(filter(lambda file: file.__contains__("frame") and file.__contains__("json"), all_files))
    data3d_files = all_files - depth_files - conf_files - frames_files - pose_files
    

    for file in depth_files:
        shutil.move(os.path.join(rawdata_path, file), os.path.join(depth_folder, file))
    
    for file in conf_files:
        shutil.move(os.path.join(rawdata_path, file), os.path.join(conf_folder, file))
        
    for file in frames_files:
        shutil.move(os.path.join(rawdata_path, file), os.path.join(frames_folder, file))
    
    for file in pose_files:
        shutil.move(os.path.join(rawdata_path, file), os.path.join(pose_folder, file))
    
    for file in data3d_files:
        shutil.move(os.path.join(rawdata_path, file), os.path.join(data3d_folder, file))

    
    print("finished")
    