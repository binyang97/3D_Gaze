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
from Apriltag_Colmap import create_geometry_at_points, visualize_2d




class ScannerApp_Reader:
    def __init__(self, datapath) -> None:
        self.datapath = datapath
        


if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":  
    # linux
        data_path  = ""
    elif platform == "win32":
    # Windows...
        data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Test2"

    VISUALIZATION = True

    images_path = os.path.join(data_path, "frames")
    pose_path = os.path.join(data_path, "pose")
    mesh_fullpath = os.path.join(data_path, "data3d/textured_output.obj")

    

    images_files = os.listdir(images_path)

    images_files.sort()

    index = 0

    image_file = images_files[index]
    camera_param_file = image_file.replace(".jpg", ".json")

    print(image_file)
    

    img = cv2.imread(os.path.join(images_path, image_file), cv2.IMREAD_GRAYSCALE)
    with open(os.path.join(pose_path, camera_param_file), 'r') as f:
        camera_param = json.load(f)

    intrinsics = np.array(camera_param["intrinsics"]).reshape(3, 3)
    print(intrinsics)

    fxfycxcy= [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]


    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    # The real size of the tag is about 8.7 cm
    tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=0.087)

    
    ext = np.array(camera_param["cameraPoseARFrame"]).reshape(4, 4)

    tag = tags[0]
    t_tag2cam = np.array(tag.pose_t).reshape(3, 1)   
    R_tag2cam = np.array(tag.pose_R)

    R_cam2tag = R_tag2cam.T
    t_cam2tag = -R_tag2cam.T @ t_tag2cam

    Cam2Tag = np.concatenate(
                    [np.concatenate([R_cam2tag, t_cam2tag], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    tag_position_cam =t_cam2tag

    r = R.from_euler('xyz', [0, 180, -90], degrees=True)
    Additional_Rotation = r.as_matrix()
    additional_extrinsic = np.concatenate(
                    [np.concatenate([Additional_Rotation, np.zeros((3, 1))], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    
    World2Cam = ext @ additional_extrinsic
    R_world2cam = World2Cam[:3, :3]
    t_world2cam = World2Cam[:3, 3].reshape(3, 1)
    R_cam2world = R_world2cam.T
    t_cam2world = -R_cam2world @ t_world2cam

    tag_position_world = R_cam2world @ tag_position_cam + t_cam2world

    tag_position_world = -tag_position_world.reshape(-1)

    print(tag_position_world)


    
    

    # x, y, z axis will be rendered as red, green, and blue

    # The camera coordinates in apriltag and 3d-scanner app are set differently 
    # So there is an additional Rotation that has to be applied in transformation
    '''
    Apriltag coordinate                     
            z  
            |         
            . y -------- .
            |/    ip    /
            o -------- x

    3D Scanner App Coordinate

             x -------- .
            /   ip     /
           o -------- y
           |
           .
           |
           z
        ''' 
    if VISUALIZATION:
        mesh = o3d.io.read_triangle_mesh(mesh_fullpath, True)
        
        coordinate = mesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))

        coordinate.transform(Cam2Tag @ World2Cam)
        print(ext[:3, 3])

        center_vis = create_geometry_at_points([ext[:3, 3], tag_position_world], color = [1, 0, 0], radius=0.1)

        o3d.visualization.draw_geometries([mesh, coordinate, center_vis])


    





    