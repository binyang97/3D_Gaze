# The file is used to evaluate the estimated camera pose
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R
import math as m
def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix


def quaternion_to_euler(w, x, y, z):

    t0 = 2 * (w * x + y * z)
    t1 = 1 - 2 * (x * x + y * y)
    X = m.atan2(t0, t1)
    t2 = 2 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = m.asin(t2)

    t3 = 2 * (w * z + x * y)
    t4 = 1 - 2 * (y * y + z * z)
    Z = m.atan2(t3, t4)

    return np.array([X, Y, Z])

def quat_to_euler(Q):
    
    euler_angles = R.from_quat(Q)
    return euler_angles  

def position_error(T_e, T_gt):

    return np.linalg.norm(T_e- T_gt)


def rotation_error(R_e, R_gt):
    
    return (180/pi) * np.arccos((np.trace(np.matmul(R_e, R_gt.T))-1)/2)

def transform_to_world(R_1_gt, T_1_gt, R_rel, T_rel):
    R_est = R_rel @ R_1_gt
    T_add = R_rel@T_1_gt
    T_est = T_rel + T_add.reshape(3, 1)

    # R_est = R_rel @ R_1_gt 
    # T_est = T_rel + R_rel@T_1_gt

    return R_est, T_est

if __name__ == "__main__":
    r1 = R.from_euler('z', 45, degrees=True)
    R1 = r1.as_matrix()


    r2 = R.from_euler('z', 30, degrees=True)
    R2 = r2.as_matrix()


    T1 = np.array([1, 1.2, 1.5]).reshape(3, 1)
    T2 = np.array([0.8, 4.3, 3]).reshape(3, 1)
    # print(rotation_error(R1, R2))
    # print(position_error(T1, T2))


    extrinsic1 = np.concatenate(
                        [np.concatenate([R1, T1], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    extrinsic2 = np.concatenate(
                        [np.concatenate([R2, T2], axis=1), np.array([[0, 0, 0, 1]])], axis=0)


    res1 = extrinsic2 @ extrinsic1

    res2_rot, res2_tran = transform_to_world(R1, T1, R2, T2)


    print(res1)
    print(res2_rot)
    print(res2_tran)


    quat = [0.014406197300745461, -0.0015940703021185547, 0.99712831150560799, 0.074330685542238983]
    quat_scipy = [-0.0015940703021185547, 0.99712831150560799, 0.074330685542238983, 0.014406197300745461]

    euler1 = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])

    euler2 = quat_to_euler(quat_scipy)

    print(euler1)
    print(euler2.as_euler('xyz'))