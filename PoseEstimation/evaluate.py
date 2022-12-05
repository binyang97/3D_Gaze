# The file is used to evaluate the estimated camera pose
import numpy as np
from math import pi

def position_error(T_e, T_gt):

    return np.linalg.norm(T_e- T_gt)


def rotation_error(R_e, R_gt):
    
    return (180/pi) * np.arccos((np.trace(np.cross(R_e, R_gt.T))-1)/2)


if __name__ == "__main__":
    T_e = np.ones(3)
    T_gt = np.zeros(3)

    R_e = np.eye(3)
    R_e[:, 2] = 2

    R_gt = np.eye(3)


    print(position_error(T_e, T_gt))
    print(rotation_error(R_e, R_gt))