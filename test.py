import numpy as np


point_cloud = np.load('/home/biyang/Documents/3D_Gaze/dataset/point_cloud.npz', allow_pickle=True)

#print(list(point_cloud.keys()))
print(point_cloud['conferenceRoom_1'][1][0].shape)


