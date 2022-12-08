import numpy as np
import os
from load_ARSceneData import read_traj
from evaluate_ARSceneData import rewrite_image_txt
from glob import glob
from sys import platform
from scipy.spatial.transform import Rotation as R
import re



if __name__ == "__main__":

    ARKitSceneDataID = "41069042"
    if platform == "linux" or platform == "linux2":  
    # linux
        #txt_filepath= '/home/biyang/Documents/3D_Gaze/Colmap/output/images.txt'
        txt_filepath = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/output/*/images.txt')
        txt_filepath.sort()

        # ground_truth
        traj_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.traj')[0]


    VISUALIZATION = False
    TRANSFORM = True
    image_id, camera_params, _, _= rewrite_image_txt(txt_filepath[0])

    for path in txt_filepath[1:]:
        image_id_extend, camera_params_extend, points_2D_extend, point3D_IDs_extend = rewrite_image_txt(path)

        image_id = image_id + image_id_extend
        camera_params = np.vstack([camera_params, camera_params_extend])

    pose_gt = read_traj(traj_path)


    marker1 = '_'
    marker2 = '.png'
    regexPattern = marker1 + '(.+?)' + marker2

    GroundTruth = []
    Estimation = []

    for i, name in enumerate(image_id):
        data_gt = []  # timestamp tx ty tz qx qy qz qw
        data_est = []  # timestamp tx ty tz qx qy qz qw
        name = name.split("/")[-1]
        ts = re.search(regexPattern, name).group(1)

        frame_id = ts
        if frame_id in pose_gt.keys():
            gt = pose_gt[frame_id]
        else:
            if str(float(frame_id)-0.001) in pose_gt.keys():
                gt = pose_gt[str(float(frame_id)-0.001)]
            elif str(float(frame_id)+0.001) in pose_gt.keys():
                gt = pose_gt[str(float(frame_id)+0.001)]
        gt = np.array(gt)
        rot_gt = gt[:3, :3]
        tran_gt = gt[:3, 3]


        #### This one is manually given and roughly estimated. ToDo: estimate the point cloud scale ratio with alighment tool (open3d)
        ### Here no change, because I used the relative pose for alignment
        scaling_factor = 0.3857

        if TRANSFORM:
            Translation = tran_gt.reshape(3, 1)
            Rotation = rot_gt
            tran_gt = np.matmul(-Rotation.T, Translation)
            rot_gt = Rotation.T
            tran_gt = tran_gt.reshape(-1)

        est = camera_params[i] # QW, QX, QY, QZ, TX, TY, TZ
        data_est = [float(ts), est[4], est[5], est[6], est[1], est[2], est[3], est[0]]

        quaternion_gt = R.from_matrix(rot_gt).as_quat() # QX, QY, QZ, QW
        data_gt = [float(ts), tran_gt[0], tran_gt[1], tran_gt[2], quaternion_gt[0], quaternion_gt[1], quaternion_gt[2], quaternion_gt[3]]


        GroundTruth.append(data_gt)
        Estimation.append(data_est)


    output_path = '/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/evaluation'
    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)

    with open(os.path.join(output_path, 'stamped_groundtruth.txt'), 'w') as fp:
        for item in GroundTruth:
            # write each item on a new line
            line = " ".join([str(elem) for elem in item])
            fp.write(line + "\n")
        print('Done')

    with open(os.path.join(output_path, 'stamped_traj_estimate.txt'), 'w') as fp:
        for item in Estimation:
            # write each item on a new line
            line = " ".join([str(elem) for elem in item])
            fp.write(line + "\n")
        print('Done')

        




        

        







    