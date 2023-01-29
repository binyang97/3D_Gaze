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
def quat_to_euler(Q, colmap = True):

    # Colmap has order qw, qx, qy, qz
    # But scipy has the order qx, qy, qz, qw
    if colmap:
        Q_reorder = np.array([Q[1], Q[2], Q[3], Q[0]])
        euler_angles = R.from_quat(Q_reorder)
    else:
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
    R_diff = R2 @ R1.T
    #T_diff = R1_inv @ (T2 - T1)
    T_diff = T2 - (R2@R1.T)@T1 
    T_diff = T_diff.reshape(3, 1)
    
    return R_diff, T_diff

# R_1_gt and T_1_gt are the ground truth transformation matrix w.r.t the world coordinate
# We combine the two transformation matrix to get the estimated results from SfM 
# order of transformation is: world->cam1 (camera of first frame) -> target frame
def transform_to_world(R_1_gt, T_1_gt, R_rel, T_rel):
    R_est = R_rel @ R_1_gt
    T_est = T_rel + R_rel@T_1_gt

    # R_est = R_rel @ R_1_gt 
    # T_est = T_rel + R_rel@T_1_gt

    return R_est, T_est
    
if __name__ == "__main__":
    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
    # linux
        #txt_filepath= '/home/biyang/Documents/3D_Gaze/Colmap/output/images.txt'
        txt_filepath = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/output/*/images.txt')
        txt_filepath.sort()
    elif platform == "win32":
    # Windows...
        txt_filepath = glob('D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/Output/images.txt')

    VISUALIZATION = True
    SCALEBACK = True
    image_id, camera_params, points_2D, point3D_IDs = rewrite_image_txt(txt_filepath[0])
    print(txt_filepath[0])

    # for path in txt_filepath[1:]:
    #     image_id_extend, camera_params_extend, points_2D_extend, point3D_IDs_extend = rewrite_image_txt(path)

    #     image_id = image_id + image_id_extend
    #     camera_params = np.vstack([camera_params, camera_params_extend])
    #     point3D_IDs = np.hstack([point3D_IDs,  point3D_IDs_extend])
    #     points_2D = np.concatenate([points_2D, points_2D_extend], axis = 0)

    #print(len(point3D_IDs))
    #print(points_2D[0][:10]
    camera_param_1 = camera_params[0]
    euler_angles_1 = quat_to_euler(camera_param_1[:4])
    rotation_1 = euler_angles_1.as_matrix()
    translation_1 = camera_param_1[4:].reshape(3, 1)

    
    #print(transform_to_frist_frame(rotation_1, translation_1, rotation_1, translation_1,))

    rotation_relative = [np.eye(3)]
    translation_relative = [np.zeros((3, 1))]
    for i, camera_param in enumerate(camera_params):
        if i == 0:
            pass
        else:
            euler_angles = quat_to_euler(camera_param[:4])
            rotation_matrix = euler_angles.as_matrix()
            translation_matrix = camera_param[4:].reshape(3, 1)

            R_new, T_new = transform_to_frist_frame(rotation_1, translation_1, rotation_matrix, translation_matrix)
            rotation_relative.append(R_new)
            translation_relative.append(T_new)

    if platform == "linux" or platform == "linux2":  
        traj_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.traj')[0]
        mesh_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.ply')[0]
        intrinsic_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.pincam')[0]

    elif platform == "win32":
        traj_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/lowres_wide.traj"
        mesh_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/40777060_3dod_mesh.ply"
        intrinsic_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/40777060_98.764.pincam"

    pose_gt, mesh_gt, intrinsics = LoadARSceneData(traj_path, mesh_path, intrinsic_path)

    # frame_1_id = image_id[0][-11:-4]
    # pose_1_gt = pose_gt[frame_1_id]
    # Need to align the ground truth to estimated result
    marker1 = '_'
    marker2 = '.png'
    regexPattern = marker1 + '(.+?)' + marker2
    pose_gt_reorder = []
    for name in image_id:
        name = name.split("/")[-1]
        frame_id = re.search(regexPattern, name).group(1)
        if frame_id in pose_gt.keys():
            pose_frame_gt = pose_gt[frame_id]
        else:
            if str(float(frame_id)-0.001) in pose_gt.keys():
                pose_frame_gt = pose_gt[str(float(frame_id)-0.001)]
            elif str(float(frame_id)+0.001) in pose_gt.keys():
                pose_frame_gt = pose_gt[str(float(frame_id)+0.001)]
        pose_gt_reorder.append(pose_frame_gt)

    pose_gt_reorder = np.array(pose_gt_reorder)

    # # Write a txt file for model_aligner with colmap 
    # output_path = '/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID
    # with open(os.path.join(output_path, 'ref_images.txt'), 'w') as fp:
    #     for i, item in enumerate(pose_gt_reorder):
    #         # write each item on a new line
    #         ref_image_data = list(item[:3,3])
    #         ref_image_data.insert(0, image_id[i])
    #         line = " ".join([str(elem) for elem in ref_image_data])
    #         fp.write(line + "\n")
    #     print('Done')

    # Conver the estimated relative pose to world coodinate with help of gt pose of the first frame
    
    # ARKitScenes provides the ground truth pose in form of camera --> world
    # Colmap outputs pose in form of world --> camera
    # The pose needs to be utilized in the same reference coordinate
    for i, pose in enumerate(pose_gt_reorder):
        Translation = pose[:3, 3].reshape(3, 1)
        Rotation = pose[:3, :3]
        T_inv = np.matmul(-Rotation.T, Translation)
        R_inv = Rotation.T
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


    # Find the scale using the distance between cam position
    came_id = [1, int(len(image_id) / 4), int(len(image_id) / 2), int(3*len(image_id) / 4), -2]

    scale_accumulated = 0
    for id in came_id:
        distance_reconstruction = np.linalg.norm(translation_estimate[0] - translation_estimate[id])
        translation_gt = np.array(pose_gt_reorder[id][:3, 3]).reshape(3, 1)
        distance_gt = np.linalg.norm(translation_1_gt -translation_gt)

        scale_accumulated += distance_gt / distance_reconstruction
    
    scale_ratio = scale_accumulated / len(came_id)

    print("Scale ratio is ", scale_ratio)

    for i in range(len(translation_estimate)):
        translation_estimate[i] = translation_estimate[i] * scale_ratio

    # Calculate the Error
    rot_error = []
    tran_error = []
    for rot_est, tran_est, pose in zip(rotation_estimate, translation_estimate, pose_gt_reorder):
        rot_error.append(rotation_error(rot_est, pose[:3, :3]))
        tran_error.append(position_error(tran_est, pose[:3, 3].reshape(3, 1)))

    print(rot_error[:30])
    print(tran_error[:10])
    #print(mesh_gt.vertices.shape)
    # Visualization 

    # est_extrinsic = np.concatenate(
    #                     [np.concatenate([rotation_estimate[1], translation_estimate[1]], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    # print(pose_gt_reorder[1])
    # print(est_extrinsic)

    print(sum(np.array(rot_error)>4))
    print(sum(np.array(tran_error)>1))
    if VISUALIZATION:
        bounds = mesh_gt.bounding_box.bounds
        corners = trimesh.bounds.corners(bounds)

        #np.random.seed(1)
        num_vis = 10
        vis_choices = np.random.choice(len(image_id), num_vis)

        visualizer = SimpleCameraPoseVisualizer([-10, 10], [-10, 10], [-5, 5])

        for index in vis_choices:
            est_extrinsic = np.concatenate(
                        [np.concatenate([rotation_estimate[index], translation_estimate[index]], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

            visualizer.extrinsic2pyramid(est_extrinsic, plt.cm.rainbow(index/len(image_id)), focal_len_scaled = 0.5, aspect_ratio=0.5, alpha = 0.6)
            visualizer.extrinsic2pyramid(pose_gt_reorder[index], plt.cm.rainbow(index/len(image_id)), focal_len_scaled = 0.5, aspect_ratio=0.5, alpha = 0.1)

        #visualizer.extrinsic2pyramid(est_extrinsic, 'r', 10)

        visualizer.customize_legend(vis_choices)
        visualizer.colorbar(len(image_id))
        visualizer.add_mesh_bbox([corners])

        visualizer.show()



