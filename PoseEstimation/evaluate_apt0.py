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
from evaluate_ARSceneData import rewrite_image_txt, transform_to_frist_frame, transform_to_world, quat_to_euler
import os


if __name__ == "__main__":
    ARKitSceneDataID = "41069042"
    if platform == "linux" or platform == "linux2":  
    # linux
        #txt_filepath= '/home/biyang/Documents/3D_Gaze/Colmap/output/images.txt'
        txt_filepath = glob('/home/biyang/Documents/3D_Gaze/Colmap/apt0/output/*/images.txt')
        txt_filepath.sort()
    elif platform == "win32":
    # Windows...
        txt_filepath = '::::'

    VISUALIZATION = True
    image_id, camera_params, points_2D, point3D_IDs = rewrite_image_txt(txt_filepath[0])

    for path in txt_filepath[1:]:
        image_id_extend, camera_params_extend, points_2D_extend, point3D_IDs_extend = rewrite_image_txt(path)

        image_id = image_id + image_id_extend
        camera_params = np.vstack([camera_params, camera_params_extend])
        point3D_IDs = np.hstack([point3D_IDs,  point3D_IDs_extend])
        points_2D = np.concatenate([points_2D, points_2D_extend], axis = 0)
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
            rotation_matrix = rotation_matrix.T
            translation_matrix = (-rotation_matrix) @ translation_matrix

            R_new, T_new = transform_to_frist_frame(rotation_1, translation_1, rotation_matrix, translation_matrix)
            rotation_relative.append(R_new)
            translation_relative.append(T_new)

    

    # Load GT data
    K = np.array([[582.871, 0, 320],
            [0, 582.871, 240],
            [0, 0, 1]])

    rotation_gt_reorder = []
    transolation_gt_reorder = []
    traj_folder = '/home/biyang/Documents/3D_Gaze/Colmap/apt0/gt/traj/'

    

    for name in image_id:
        cam_name = name.replace('jpg', 'cam')
        traj_path = os.path.join(traj_folder, cam_name)
        cam = np.loadtxt(traj_path, max_rows = 1)
        T = np.array([[cam[0]], [cam[1]], [cam[2]]])
        R = np.array([[cam[3], cam[4], cam[5]],
                [cam[6], cam[7], cam[8]],
              [cam[9], cam[10], cam[11]]])

        rotation_gt_reorder.append(R)
        transolation_gt_reorder.append(T)

    rotation_gt_reorder = np.array(rotation_gt_reorder)
    transolation_gt_reorder = np.array(transolation_gt_reorder)

    mesh_gt = trimesh.load_mesh('/home/biyang/Documents/3D_Gaze/Colmap/apt0/gt/apt0.ply')

    rotation_1_gt = rotation_gt_reorder[0]
    translation_1_gt = transolation_gt_reorder[0]

    # print(image_id[0], image_id[1])
    # _, dist = transform_to_frist_frame(rotation_gt_reorder[0], transolation_gt_reorder[0], rotation_gt_reorder[1], transolation_gt_reorder[1])

    # print(np.linalg.norm(dist))
    # print(np.linalg.norm(transolation_gt_reorder[0] - transolation_gt_reorder[1]))
    
    rotation_estimate = []
    translation_estimate = []
    
    for rot_gt in rotation_gt_reorder:
        print(np.linalg.det(rot_gt))

    for (rel_rot, rel_tran) in zip(rotation_relative, translation_relative):
        rotation_est, translation_est = transform_to_world(rotation_1_gt, translation_1_gt, rel_rot, rel_tran) 
        rotation_estimate.append(rotation_est)
        translation_estimate.append(translation_est)

    rot_error = []
    tran_error = []
    for rot_est, tran_est, rot_gt, tran_gt in zip(rotation_estimate, translation_estimate, rotation_gt_reorder, transolation_gt_reorder):   
        rot_error.append(rotation_error(rot_est, rot_gt))
        tran_error.append(position_error(tran_est, tran_gt))
    print(sum(np.array(rot_error)>5))
    print(sum(np.array(tran_error)>1))

    print(rot_error[-20:])
    print(tran_error[-20:])
    if VISUALIZATION:
        bounds = mesh_gt.bounding_box.bounds
        corners = trimesh.bounds.corners(bounds)

        #np.random.seed(1)
        num_vis = 5
        vis_choices = np.random.choice(len(image_id), num_vis)

        visualizer = SimpleCameraPoseVisualizer([-10, 10], [-10, 10], [-5, 5])

        for index in vis_choices:
            est_extrinsic = np.concatenate(
                        [np.concatenate([rotation_estimate[index], translation_estimate[index]], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

            gt_extrinsic = np.concatenate(
                        [np.concatenate([rotation_gt_reorder[index], transolation_gt_reorder[index]], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

            visualizer.extrinsic2pyramid(est_extrinsic, plt.cm.rainbow(index/len(image_id)), focal_len_scaled = 0.5, aspect_ratio=0.5, alpha = 0.6)
            visualizer.extrinsic2pyramid(gt_extrinsic, plt.cm.rainbow(index/len(image_id)), focal_len_scaled = 0.5, aspect_ratio=0.5, alpha = 0.1)

        #visualizer.extrinsic2pyramid(est_extrinsic, 'r', 10)

        visualizer.customize_legend(vis_choices)
        visualizer.colorbar(len(image_id))
        visualizer.add_mesh_bbox([corners])

        visualizer.show()
