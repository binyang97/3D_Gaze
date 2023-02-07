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
        data_path  = "/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/Test2"
    elif platform == "win32":
    # Windows...
        data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1-dataset2"

    # Getting the Visualization
    VISUALIZATION = True

    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )


    images_path = os.path.join(data_path, "frames")
    pose_path = os.path.join(data_path, "pose")
    mesh_fullpath = os.path.join(data_path, "data3d/textured_output.obj")
    depth_path = os.path.join(data_path, "depth")

    

    images_files = os.listdir(images_path)

    images_files.sort()

    tag_points_3d = []
    projected_points_3d = []
    r = R.from_euler('xyz', [0, 180, -90], degrees=True)
    Additional_Rotation = r.as_matrix()

    additional_rotation = np.concatenate(
                    [np.concatenate([Additional_Rotation, np.zeros((3,1))], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    for i, image_file in enumerate(images_files):
        if i%10 != 0:
            continue


        camera_param_file = image_file.replace(".jpg", ".json")
        depth_file = image_file.replace("frame", "depth")
        #depth_file = depth_file.replace(".jpg", ".png")

        #print(depth_file)


        img = cv2.imread(os.path.join(images_path, image_file), cv2.IMREAD_GRAYSCALE)
        #depth = cv2.imread(os.path.join(depth_path, depth_file))
        #print(np.unique(depth))

        img_height, img_width = img.shape
        with open(os.path.join(pose_path, camera_param_file), 'r') as f:
            camera_param = json.load(f)

        intrinsics = np.array(camera_param["intrinsics"]).reshape(3, 3)

        fxfycxcy= [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]


        projectionMatrix = np.array(camera_param["projectionMatrix"]).reshape(4, 4)



    
        # The real size of the tag is about 8.7 cm
        tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=0.087)

        if len(tags) == 0:
            continue
        # cam2world
        ext = np.array(camera_param["cameraPoseARFrame"]).reshape(4, 4)
        
        Cam2World = ext @ additional_rotation

        for tag in tags:
            tag_position_cam = np.concatenate((np.array(tag.pose_t), np.ones((1, 1))), axis = 0 )
            tag_position_world = Cam2World@tag_position_cam
            
            tag_points_3d.append(tag_position_world[:3])

            tag_center = np.concatenate((np.array(tag.center), np.ones(2))).reshape(4, 1)
            tag_center[0] = tag_center[0]/img_height
            tag_center[1] = tag_center[1]/img_width

            tag_center_3d = projectionMatrix @ tag_center

            projected_points_3d.append(tag_center_3d[:3])

    print(projected_points_3d)        
    



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
        #
        # mesh.transform(Cam2World)
        
        coordinate = mesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        

        tag_points = create_geometry_at_points(tag_points_3d, color = [1, 0, 0], radius=0.05)
        projected_points = create_geometry_at_points(projected_points_3d, color = [0, 1, 0], radius = 0.05)


        o3d.visualization.draw_geometries([mesh, coordinate, tag_points, projected_points])


    





    