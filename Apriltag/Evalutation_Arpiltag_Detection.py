# from Apriltag_Colmap import *
# from Apriltag_Test_CameraPose import *

from sys import platform
import os
import cv2
import json
from pupil_apriltags import Detector
import open3d as o3d
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


def linear_fit(x, y, err = "Mean"):
    """For set of points `(xi, yi)`, return linear polynomial `f(x) = k*x + m` that
    minimizes the sum of quadratic errors.
    """
    meanx = sum(x) / len(x)
    meany = sum(y) / len(y)
    k = sum((xi-meanx)*(yi-meany) for xi,yi in zip(x,y)) / sum((xi-meanx)**2 for xi in x)
    m = meany - k*meanx

    sum_error = 0
    for xi, yi in zip(x,y):
        y_line = k*xi + m
        sum_error += abs(y_line - yi)
    
    if err == "Mean":
        error = sum_error / len(x)
    elif err == "Sum":
        error = sum_error

    return k, m, error

class ScannerApp_Reader:
    def __init__(self, datapath) -> None:
        self.datapath = datapath
        

def draw_axis(img, center, imgpts):
    center = tuple(center.ravel())
    img = cv2.line(img, center, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, center, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, center, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(255,20,147),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(255,20,147),3)
    return img

if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":  
    # linux
        data_path  = "/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/Apriltag1_dataset1"
        data_pi_path = "/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/office1/data_1"
    elif platform == "win32":
    # Windows...
        data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1_dataset1"
        data_pi_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\office1\data_1"

    # Getting the Visualization
    VISUALIZATION = False
    TAG_POSE_VISUALIZATION = False
    DATA = "IPHONE" # "PI" or "IPHONE"

    Evaluation = {}

    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    if DATA == "IPHONE":

        images_path = os.path.join(data_path, "frames")
        pose_path = os.path.join(data_path, "pose")
        mesh_fullpath = os.path.join(data_path, "data3d/textured_output.obj")
        depth_path = os.path.join(data_path, "depth")

    elif DATA == "PI":
        data_path = data_pi_path
        images_path = os.path.join(data_path, "images_undistorted")
    
    tag_sizes = np.arange(0.05, 0.11, 0.01)
    real_tag_size = 0.087


    images_files = os.listdir(images_path)

    images_files.sort()

    tag_points_3d = []

    Vis_frames = []
    if DATA == "IPHONE":
    
        img_width = 1440
        img_height = 1920

    elif DATA == "PI":
        img_width = 1088
        img_height = 1080
        print("There is no extrinsic matrix and 3d model for data recorded by PI, so there is no 3d visualization, only visulization with tag pose")

    for i, image_file in enumerate(images_files):
        # if i%5 != 0:
        #     continue
        

        img = cv2.imread(os.path.join(images_path, image_file), cv2.IMREAD_GRAYSCALE)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if DATA == "IPHONE":
        
            camera_param_file = image_file.replace(".jpg", ".json")
            with open(os.path.join(pose_path, camera_param_file), 'r') as f:
                camera_param = json.load(f)

            intrinsics = np.array(camera_param["intrinsics"]).reshape(3, 3)

            projectionMatrix = np.array(camera_param["projectionMatrix"]).reshape(4, 4)

        elif DATA == "PI":
            intrinsics = np.array([[766.2927454396544, 0.0, 543.6272327745995],
                                [0.0, 766.3976103393867, 566.0580149497666],
                                [0.0, 0.0, 1.0]])

            projectionMatrix = np.eye(4)
            
        fxfycxcy= [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
        principal_point = np.array([intrinsics[0, 2], intrinsics[1, 2]])
        focal_length = (intrinsics[0, 0] + intrinsics[1, 1]) / 2
    
        # The real size of the tag is about 8.7 cm
        tags_real = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=real_tag_size)
        if len(tags_real) == 0:
            continue
        
        alpha_real = []
        for tag_real in tags_real:
            tag_id = tag_real.tag_id
            if tag_id not in Evaluation.keys():
                Evaluation[tag_id] = {"Real_Distance": [],
                                        "Frame_id": [],
                                        "Error_Stability": [],
                                        "Error_Accuracy": [],
                                        "angle_x": [],
                                        "angle_y": [],
                                        "angle_z": []}

                # Add the real distance
            # Project the line onto the plane which is parallel to the image plane

            R_tag2cam = np.array(tag_real.pose_R)
            t_tag2cam = np.array(tag_real.pose_t)

            r = R.from_matrix(R_tag2cam)
            euler_angles = r.as_euler('xyz', degrees=True)

            Evaluation[tag_id]["Real_Distance"].append(np.linalg.norm(t_tag2cam))
            Evaluation[tag_id]["Frame_id"].append(image_file)
            Evaluation[tag_id]["angle_x"].append(euler_angles[0])
            Evaluation[tag_id]["angle_y"].append(euler_angles[1])
            Evaluation[tag_id]["angle_z"].append(euler_angles[2])

            alpha_real_tag_mean = 0
            for i, tag_corner in enumerate(tag_real.corners):
             # 0: right bottom, 1: right top, 2: left top, 3: left bottom
                tag_center = np.array(tag_real.center)
                if i == 0:
                    tag_corner_tag_coord = np.array([[1], [1], [0]]) * (real_tag_size/2)
                elif i == 1:
                    tag_corner_tag_coord = np.array([[1], [-1], [0]]) * (real_tag_size/2)
                elif i == 2:
                    tag_corner_tag_coord = np.array([[-1], [-1], [0]]) * (real_tag_size/2)
                elif i == 3:
                    tag_corner_tag_coord = np.array([[-1], [1], [0]]) * (real_tag_size/2)
                tag_corner_cam_coord = R_tag2cam @ tag_corner_tag_coord + t_tag2cam
                tag_center_cam_coord = t_tag2cam

                delta_l = math.sqrt(math.pow((tag_corner_cam_coord[0]*tag_center_cam_coord[2]/tag_corner_cam_coord[2] - tag_center_cam_coord[0]), 2) +
                                    math.pow((tag_corner_cam_coord[1]*tag_center_cam_coord[2]/tag_corner_cam_coord[2] - tag_center_cam_coord[1]), 2))

                
                delta_d = math.sqrt(math.pow(focal_length, 2) + math.pow(np.linalg.norm(tag_center - principal_point), 2))
                delta_uv_tag = np.linalg.norm(tag_corner-tag_center)
                alpha_real_tag_mean += (delta_l / real_tag_size) * (delta_d / delta_uv_tag)
            alpha_real.append(alpha_real_tag_mean / 4)

            
        
        true_distances = []
        for tag_size in tag_sizes:   
            
            tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=tag_size)
            
            # Test with only one tag
            for tag_test in tags:
                
                tag_position = np.array(tag_test.pose_t)
                tag_cam_distance = np.linalg.norm(tag_position)
                true_distances.append(tag_cam_distance)

        true_distances = np.array(true_distances).reshape(-1, len(tags))
        print(true_distances.shape)
        for i, tag in enumerate(tags_real):
            alpha_test, _, err = linear_fit(tag_sizes, true_distances[:,i])
            alpha_error = abs(alpha_real[i] - alpha_test)
            


            Evaluation[tag.tag_id]["Error_Stability"].append(err)
            Evaluation[tag.tag_id]["Error_Accuracy"].append(alpha_error)
            
        

    with open(r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/evaluation_apriltag_detection_Iphone.json", "w") as outfile:
         json.dump(Evaluation, outfile, indent=4)


        
        





        

        
        # plt.plot(tag_sizes, true_distances, "bo")
        # plt.xlabel("tag size")
        # plt.ylabel("true distance")
        # plt.title(image_file + " and tag id: " + str(tag_id))
        # plt.show()
        
        
        # cam2worl



    





    