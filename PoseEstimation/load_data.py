import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from trimesh.scene.cameras import Camera
import math
from load_ARSceneData import LoadARSceneData
from sys import platform
from evaluate import position_error, rotation_error
from camera_pose_visualizer import CameraPoseVisualizer
import matplotlib as plt
import random
# load txt file

#cameras_data = np.loadtxt('./Output/images.txt')

# Rewrite the data in images.txt (which outputs from the colmap)
# Split to camera pose data <--> image name, points2D <--> point ID

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
                
 # convert the quartenion to euler angles
def quat_to_euler(Q):
    
    euler_angles = R.from_quat(Q)
    return euler_angles  


def eulerAnglesToRotationMatrix(theta):
    """Euler rotation matrix with clockwise logic.
    Rotation
    Args:
        theta: list of float
            [theta_x, theta_y, theta_z]
    Returns:
        R: np.array (3, 3)
            rotation matrix of Rz*Ry*Rx
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))
    return 


     
# In usual SfM process, the world coordinate is set to the camera location of first frame
# But as proposed in the issue of colmap, the coordinate origin is drifted after applying the bundle adjustment
# So we have to munually transform back to the first frame for all other frames
def transform_to_frist_frame(R1, T1, R2, T2):
    R1_inv = np.linalg.inv(R1)
    R_diff = R1_inv @ R2
    T_diff = R1_inv @ (T2 - T1)
    T_diff = T_diff.reshape(3,1)

    return R_diff, T_diff

# R_1_gt and T_1_gt are the ground truth transformation matrix w.r.t the world coordinate
# We combine the two transformation matrix to get the estimated results from SfM 
# order of transformation is: world->cam1 (camera of first frame) -> target frame
def transform_to_world(R_1_gt, T_1_gt, R_rel, T_rel):
    R_est = R_1_gt @ R_rel
    T_est = T_1_gt + R_1_gt @ T_rel

    return R_est, T_est
    
if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":  
    # linux
        txt_filepath= '/home/biyang/Documents/3D_Gaze/Colmap/output/images.txt'
    elif platform == "win32":
    # Windows...
        txt_filepath = 'D:/Documents/Semester_Project/Colmap_Test/Output/images.txt'

    VISUALIZATION = True

    image_id, camera_params, points_2D, point3D_IDs = rewrite_image_txt(txt_filepath)

    #print(len(point3D_IDs))
    #print(points_2D[0][:10]
    camera_param_1 = camera_params[0]
    euler_angles_1 = quat_to_euler(camera_param_1[:4])
    rotation_1 = euler_angles_1.as_matrix()
    translation_1 = camera_param_1[4:].T
    
    rotation_relative = [np.eye(3)]
    translation_relative = [np.zeros((3, 1))]
    for i, camera_param in enumerate(camera_params):
        if i == 0:
            pass
        else:
            euler_angles = quat_to_euler(camera_param[:4])
            rotation_matrix = euler_angles.as_matrix()
            translation_matrix = camera_param[4:].T

            R_new, T_new = transform_to_frist_frame(rotation_1, translation_1, rotation_matrix, translation_matrix)
            rotation_relative.append(R_new)
            translation_relative.append(T_new)

    if platform == "linux" or platform == "linux2":  
        traj_path = '/home/biyang/Documents/3D_Gaze/Colmap/gt/lowres_wide.traj'
        mesh_path = "/home/biyang/Documents/3D_Gaze/Colmap/gt/40777060_3dod_mesh.ply"
        intrinsic_path = "/home/biyang/Documents/3D_Gaze/Colmap/gt/40777060_98.764.pincam"

    elif platform == "win32":
        traj_path = "D:/Documents/Semester_Project/Colmap_Test/GT/lowres_wide.traj"
        mesh_path = "D:/Documents/Semester_Project/Colmap_Test/GT/40777060_3dod_mesh.ply"
        intrinsic_path = "D:/Documents/Semester_Project/Colmap_Test/GT/40777060_98.764.pincam"

    pose_gt, mesh_gt, intrinsics = LoadARSceneData(traj_path, mesh_path, intrinsic_path)


    # frame_1_id = image_id[0][-11:-4]
    # pose_1_gt = pose_gt[frame_1_id]
    
    # Need to align the ground truth to estimated result
    pose_gt_reorder = []
    for image in image_id:
        frame_id = image[-11:-4]
        if frame_id in pose_gt.keys():
            pose_frame_gt = pose_gt[frame_id]
        else:
            if str(float(frame_id)-0.001) in pose_gt.keys():
                pose_frame_gt = pose_gt[str(float(frame_id)-0.001)]
            elif str(float(frame_id)+0.001) in pose_gt.keys():
                pose_frame_gt = pose_gt[str(float(frame_id)+0.001)]
        pose_gt_reorder.append(pose_frame_gt)


    # Conver the estimated relative pose to world coodinate with help of gt pose of the first frame
    pose_gt_reorder = np.array(pose_gt_reorder)
    # ARKitScenes provides the ground truth pose in form of camera --> world
    # Colmap outputs pose in form of world --> camera
    # The pose needs to be utilized in the same reference coordinate
    for i, pose in enumerate(pose_gt_reorder):
        T = pose[:3, 3].reshape(3, 1)
        R = pose[:3, :3]
        T_inv = np.matmul(-R.T, T)
        R_inv = R.T
        RT_inv = np.concatenate(((R_inv, T_inv)), axis=-1)
        world2cam = np.vstack((RT_inv, np.array([0, 0, 0, 1])))
        pose_gt_reorder[i] = world2cam

    pose_1_gt = pose_gt_reorder[0]

    rotation_1_gt = np.array(pose_1_gt[:3, :3])
    translation_1_gt = np.array(pose_1_gt[:3, 3]).reshape(3, 1)
    
    rotation_estimate = []
    translation_estimate = []
     
    for (rel_rot, rel_tran) in zip(rotation_relative, translation_relative):
        rotation_est, translation_est = transform_to_world(rotation_1_gt, translation_1_gt, rel_rot, rel_tran) 
        rotation_estimate.append(rotation_est)
        translation_estimate.append(translation_est)


    # Calculate the Error
    rot_error = []
    tran_error = []
    for (rot_est, tran_est, pose_gt) in zip(rotation_estimate, translation_estimate, pose_gt_reorder):
        rot_error.append(rotation_error(rot_est, pose_gt[:3, :3]))
        tran_error.append(position_error(tran_est, pose_gt[:3, 3]))


    print(rot_error[:10])
    print(tran_error[:10])
    #print(mesh_gt.vertices.shape)
    # Visualization 

    if VISUALIZATION:
        bounds = mesh_gt.bounding_box.bounds
        corners = trimesh.bounds.corners(bounds)

        num_vis = 5
        vis_choices = np.random.choice(len(image_id), num_vis)

        visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [-5, 5])

        for index in vis_choices:
            est_extrinsic = np.concatenate(
                        [np.concatenate([rotation_estimate[index], translation_estimate[index]], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

            visualizer.extrinsic2pyramid(est_extrinsic, plt.cm.rainbow(index/len(image_id)), 1, alpha = 0.6)
            visualizer.extrinsic2pyramid(pose_gt_reorder[index], plt.cm.rainbow(index/len(image_id)), 1, alpha = 0.1)

        #visualizer.extrinsic2pyramid(est_extrinsic, 'r', 10)

        visualizer.customize_legend(vis_choices)
        visualizer.colorbar(len(image_id))
        visualizer.add_mesh_bbox([corners])

        visualizer.show()



