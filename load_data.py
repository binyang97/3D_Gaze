import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from trimesh.scene.cameras import Camera
import math
from load_ARSceneData import LoadARSceneData
# load txt file

#cameras_data = np.loadtxt('./Output/images.txt')

# Rewrite the data in images.txt (which outputs from the colmap)
# Split to camera pose data <--> image name, points2D <--> point ID


class Visualizer(object):
    def __init__(self, camera_gt, camera_est, mesh, resolution, focal_length):
        self.model = mesh
        self.pose_est = camera_est
        self.pose_gt = camera_gt
        self.resolution = resolution
        self.focal_length = focal_length

    def visualization(self):
        scene = self.model.scene()
        camera1 = Camera(resolution=self.resolution, focal_length=self.focal_length)
        camera_marker_gt = trimesh.creation.camera_marker(self.gt, marker_height=0.4, origin_size=None)

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
    T_est = T_1_gt + T_rel

    return R_est, T_est
    
if __name__ == "__main__":
    #txt_filepath= '/home/biyang/Documents/3D_Gaze/Colmap/output/images.txt'
    txt_filepath = 'D:/Documents/Semester_Project/Colmap_Test/Output/images.txt'

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


    traj_path = "D:/Documents/Semester_Project/Colmap_Test/GT/lowres_wide.traj"
    mesh_path = "D:/Documents/Semester_Project/Colmap_Test/GT/40777060_3dod_mesh.ply"
    intrinsic_path = "D:/Documents/Semester_Project/Colmap_Test/GT/40777060_98.764.pincam"

    pose_gt, mesh_gt, intrinsics = LoadARSceneData(traj_path, mesh_path, intrinsic_path)


    frame_1_id = image_id[0][-11:-4]
    pose_1_gt = pose_gt[frame_1_id]
    
    # Need to align the ground truth to estimated result
    pose_gt_reorder = []
    for image in image_id:
        frame_id = image[-11:-4]
        pose_frame_gt = pose_gt[frame_id]
        pose_gt_reorder.append(pose_frame_gt)

    



    # print(rotation_relative[0])
    # print(rotation_relative[1])
    # print(translation_relative[0])
    # print(translation_relative[1])

