import trimesh
import numpy 
import os
from PIL import Image

#import scipy.io
import h5py
import numpy as np

def num2str(char_list):
    return ''.join(chr(x) for x in char_list)

f = h5py.File('/home/biyang/Documents/3D_Gaze/dataset/area_1_no_xyz/area_1/3d/pointcloud.mat','r')

point_cloud = {}

for i in range(len(np.array(f['Area_1']['Disjoint_Space']['object']))):
#for i in range(2):
    # The srting in matlab is loaded as characters in python, it needs to beconvert
    ref_name = f['Area_1']['Disjoint_Space']['name'][i][0]
    room_name = f[ref_name]
    room_name = np.array(room_name).squeeze()
    room_name = num2str(room_name)

    #room_part = {}

    ref = f['Area_1']['Disjoint_Space']['object'][i][0]
    
    sub_room_names = []
    sub_room_points = []
    for j in range(len(f[ref]['points'])):
    #for j in range(2):
        data = f[ref]['points'][j][0]
        sub_ref_name = f[ref]['name'][j][0]

        points = f[data]
        sub_room_name = f[sub_ref_name]

        points = np.array(points).transpose()
        sub_room_name = np.array(sub_room_name).squeeze()

        sub_room_name = num2str(sub_room_name)
        sub_room_names.append(sub_room_name)
        sub_room_points.append(points)

    point_cloud[room_name] = [sub_room_names, sub_room_points]


#np.savez('/home/biyang/Documents/3D_Gaze/dataset/point_cloud.npz', **point_cloud)

np.savez('/home/biyang/Documents/3D_Gaze/dataset/point_cloud.npz', **point_cloud)









# ref = f['Area_1']['Disjoint_Space']['object'][43][0]


# data = f[ref]['points'][0][0]

# points = np.array(data)

# test = f[data]

# test = np.array(test).transpose()[0]




# #name = f.get('Area_1/Disjoint_Space/object')
# #name = f.get('Area_1/Disjoint_Space/name')

# #print(np.array(name))

# ref_name = f['Area_1']['Disjoint_Space']['name'][43][0]
# name = f[ref_name]
# name = np.array(name).squeeze()
# print(name)
# str1 = ''.join(chr(x) for x in name)
# print(str1)

# #str1 = ''.join(chr(i) for i in name[:])
# #name = f[name]

# #test = data
# #obj = f[test]

# #data = [obj for obj in test[:]]
# #ref = data[0]

# #name = np.array(name)

# #data = np.array(data)

# print(test)
# #print(str1)