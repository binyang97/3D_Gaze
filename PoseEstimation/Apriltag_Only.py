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
from Apriltag_Colmap import create_geometry_at_points, visualize_2d, colorbar




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
        data_path  = "/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/Test2"
    elif platform == "win32":
    # Windows...
        data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1-dataset2"
        data_pi_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\office1\data_1"

    # Getting the Visualization
    VISUALIZATION = True
    TAG_POSE_VISUALIZATION = False
    DATA = "IPHONE" # "PI" or "IPHONE"

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
    

    images_files = os.listdir(images_path)

    images_files.sort()

    tag_points_3d = []
    projected_points_3d = []
    r = R.from_euler('xyz', [0, 180, -90], degrees=True)
    Additional_Rotation = r.as_matrix()

    additional_rotation = np.concatenate(
                    [np.concatenate([Additional_Rotation, np.zeros((3,1))], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    Vis_frames = []
    if DATA == "IPHONE":
    
        img_width = 1440
        img_height = 1920

    elif DATA == "PI":
        img_width = 1088
        img_height = 1080
        print("There is no extrinsic matrix and 3d model for data recorded by PI, so there is no 3d visualization, only visulization with tag pose")

    for i, image_file in enumerate(images_files):
        if i%10 != 0:
            continue
        

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
    
        # The real size of the tag is about 8.7 cm
        tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=0.087)

        if len(tags) == 0:
            Vis_frames.append(color_img)
            continue
        # cam2world

        if DATA == "IPHONE":
            ext = np.array(camera_param["cameraPoseARFrame"]).reshape(4, 4)      
            Cam2World = ext @ additional_rotation

        if DATA == "PI":
            Cam2World = np.eye(4)


        for tag in tags:
            tag_position_cam = np.concatenate((np.array(tag.pose_t), np.ones((1, 1))), axis = 0 )
            tag_position_world = Cam2World@tag_position_cam
            
            tag_points_3d.append(tag_position_world[:3])

            tag_center = np.concatenate((np.array(tag.center), np.ones(2))).reshape(4, 1)
            tag_center[0] = tag_center[0]/img_height
            tag_center[1] = tag_center[1]/img_width

            tag_center_3d = projectionMatrix @ tag_center

            projected_points_3d.append(tag_center_3d[:3])


            # Visualize the axis and cube on the tags
            R_tag2cam = np.array(tag.pose_R)
            t_tag2cam = np.array(tag.pose_t).reshape(3, 1)

            R_cam2tag = R_tag2cam.T
            t_cam2tag = -R_tag2cam.T @ t_tag2cam

            if TAG_POSE_VISUALIZATION:

                axis = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3) * 0.5
                #rot_vec,_ = cv2.Rodrigues(tag.pose_R)
                imgpts, _ = cv2.projectPoints(axis, R_tag2cam, t_tag2cam.reshape(-1), intrinsics, 0)
                color_img = draw_axis(color_img, tag.center.astype(int), imgpts.astype(int))


                axis_cube = np.float32([[-1,-1,0], [-1,1,0], [1,1,0], [1,-1,0],
                    [-1,-1,-2],[-1,1,-2],[1,1,-2],[1,-1,-2] ]) * 0.5*0.087


                imgpts_cube, _= cv2.projectPoints(axis_cube, R_tag2cam, t_tag2cam.reshape(-1), intrinsics, 0)

                color_img = draw_cube(color_img, imgpts_cube.astype(int))          

        if TAG_POSE_VISUALIZATION:
        
            #cv2.namedWindow("Detected tags", cv2.WINDOW_NORMAL) 
            cv2.imshow("Detected tags", cv2.resize(color_img, (720, 960)))

            k = cv2.waitKey(0)
            cv2.destroyAllWindows()

            #Vis_frames.append(color_img)

    # video_output_path = os.path.join(data_path, "video_axis.mp4")

    # fps = 10
    # video = cv2.VideoWriter(video_output_path,-1,fps,(img_width, img_height))

    # for frame in Vis_frames:
    #     video.write(frame)

    # cv2.destroyAllWindows()
    # video.release()


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
        #projected_points = create_geometry_at_points(projected_points_3d, color = [0, 1, 0], radius = 0.05)


        o3d.visualization.draw_geometries([mesh, coordinate, tag_points])


    





    