import os 
import numpy as np
import trimesh
import cv2

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)
    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)



def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix


# The trasjectory file contains the ground truth camera pose 
def read_traj(path):
    with open(path) as f:
        traj_file = f.readlines()
    # convert traj to json dict
    poses_from_traj = {}
    for line in traj_file:
        traj_timestamp = line.split(" ")[0]
        poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)[1].tolist()

    return poses_from_traj

def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

# Load the ground truth mesh
def load_mesh(path):
    return trimesh.load_mesh(path)

# Load the camera intrinsic parameters
def read_intrinsic(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    intrinsics = st2_camera_intrinsics(path)
    return intrinsics


# main function of loading data
def LoadARSceneData(traj_path, mesh_path, intrinsic_path):
    mesh = load_mesh(mesh_path)
    pose = read_traj(traj_path)
    K = read_intrinsic(intrinsic_path)

    return pose, mesh, K


if __name__ == "__main__":
    traj_path = "D:/Documents/Semester_Project/Colmap_Test/GT/lowres_wide.traj"
    mesh_path = "D:/Documents/Semester_Project/Colmap_Test/GT/40777060_3dod_mesh.ply"
    intrinsic_path = "D:/Documents/Semester_Project/Colmap_Test/GT/40777060_98.764.pincam"


    mesh = load_mesh(mesh_path)
    print(mesh)

    pose = read_traj(traj_path)
    print(pose.keys())

    K = read_intrinsic(intrinsic_path)
    print(K)

