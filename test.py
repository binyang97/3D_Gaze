# import numpy as np


# point_cloud = np.load('/home/biyang/Documents/3D_Gaze/dataset/point_cloud.npz', allow_pickle=True)

# #print(list(point_cloud.keys()))
# print(point_cloud['conferenceRoom_1'][1][0].shape)


import struct

# fileName = '/home/biyang/Documents/3D_Gaze/Colmap/dense/0/sparse/cameras.bin'

# with open(fileName, mode='rb') as file: # b is important -> binary
#     fileContent = file.read()

# #data = struct.unpack("i" * ((len(fileContent) -24) // 4), fileContent[20:-4])
# print(len(fileContent))

# data = struct.unpack('i'*(len(fileContent) // 4), fileContent)
# #data = list(fileContent)

# print(data[:20])

import Colmap_GUI.database as cdb

database_path = '/home/biyang/Documents/3D_Gaze/Colmap_GUI/database.db'
db = cdb.COLMAPDatabase.connect(database_path)

# Get the all attributes
sql_query = """SELECT name FROM sqlite_master  
  WHERE type='table';"""

tables = db.execute(sql_query)

print(tables.fetchall())

# rows = db.execute("SELECT * FROM cameras")
# camera_data = rows.fetchall()
# single_data = camera_data[0][4]
# print(len(single_data))
# data = struct.unpack('<L', single_data[:4])
# print(data)

# rows = db.execute("SELECT * FROM keypoints")
# keypoints_data = rows.fetchall()
# keypoint = keypoints_data[0][3]

# data = struct.unpack('<L', keypoint[4:8])

# print(data)


rows = db.execute("SELECT * FROM descriptors")
descriptors = rows.fetchall()

data = descriptors[0][:3]
print(data)


