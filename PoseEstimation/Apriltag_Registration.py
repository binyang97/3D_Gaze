from sys import platform
import os
import cv2
import json
from pupil_apriltags import Detector
import open3d as o3d
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from Apriltag_Colmap import create_geometry_at_points, visualize_2d, colorbar

at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )



if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":  
    # linux
        data_path  = "/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/Test2"
    elif platform == "win32":
    # Windows...
        data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1_dataset1"
        data_pi_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\office1\data_1"

        evaluation_json_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\evaluation_apriltag_detection_Iphone_a1d1.json"

    with open(evaluation_json_path, "r") as f:
        evaluation  = json.load(f)

    track_frame = {}
    for tag_id in evaluation.keys():
        distance_error = [alpha_error * 0.087 for alpha_error in evaluation[tag_id]["Error_Accuracy"]]
        distance_error = np.array(distance_error)
        
        index = np.argmin(distance_error)
        frame_id = evaluation[tag_id]["Frame_id"][index]

        track_frame[tag_id] = frame_id


    iphone_frames_path = os.path.join(data_path, "frames")
    tags_real = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=real_tag_size)
        




