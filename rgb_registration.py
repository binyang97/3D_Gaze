import trimesh
import cv2
#import mesh_raycast

import trimesh_new

import numpy as np

import matplotlib.pyplot as plt

import PIL

import time

import glob

import os

import pyembree
## Load the data
#mesh = trimesh.load_mesh("/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0.ply")

current_path = os.getcwd()

mesh = trimesh.load_mesh(current_path + "/dataset/apt0/apt0.ply")

# rgb = cv2.imread("/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0/color_00.jpg")
index = 35

#path_rgb_cam = "/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0/*"

files = glob.glob(current_path + "/dataset/apt0/apt0/*")

cam_files = []
rgb_files = []

for file in files:
    if file.__contains__('.cam'):
        cam_files.append(file)
    elif file.__contains__('color'):
        rgb_files.append(file)

cam_files.sort()
rgb_files.sort()

cam = np.loadtxt(cam_files[index], max_rows = 1)

T = np.array([[cam[0]], [cam[1]], [cam[2]]])
R = np.array([[cam[3], cam[4], cam[5]],
                [cam[6], cam[7], cam[8]],
              [cam[9], cam[10], cam[11]]])

#print(mesh.faces.shape)

# Select a rgb image as the input


# Camera Paramters

# Extrinsic Parameters (world to camera)


# Visulization 
visualize_3d = True
visualize_depth = True
visulize_rgb = True


if visulize_rgb:
    rgb = PIL.Image.open(rgb_files[index])
    rgb.show()
# T = np.array([[0.3019], [0.177447], [-0.055902]])
# R = np.array([[-0.836422, 0.427017, -0.343591],
#                 [-0.214059, 0.322595, 0.922015],
#               [0.504557, 0.844742 , -0.178419]])


RT = np.concatenate(((R, T)), axis=-1)
world2cam = np.vstack((RT, np.array([0, 0, 0, 1])))


# Transformation matrix 
T_inv = np.matmul(-R.T, T)
R_inv = R.T

RT_inv = np.concatenate(((R_inv, T_inv)), axis=-1)
cam2world = np.vstack((RT_inv, np.array([0, 0, 0, 1])))



# Intrinsic Paramters

K = np.array([[582.871, 0, 320],
            [0, 582.871, 240],
            [0, 0, 1]])

# Apply the transform to camera coordinate
mesh.apply_transform(world2cam)

# Create a scene for ray casting
scene = mesh.scene()

# any of the automatically generated values can be overridden

# set resolution, in pixels
scene.camera.resolution = [640, 480]

# set intrinsic parameters
scene.camera.K = K

# set extrinsic parameters 
#scene.camera_transform = world2cam

camera = scene.camera

# convert the camera to rays with one ray per pixel
## Rewrite the function with self-changed trimesh source codes
# Get the unit ray direction (z = -1)
vectors, pixels = trimesh_new.camera_to_rays(camera)
# Origins are refering to the camera coordinate
origins = np.zeros_like(vectors)
# Inverse the vector direction
vectors = -vectors


# do the actual ray- mesh queries (trimesh with pyembree)
# The test time is about 1 second per image with resolution 640 x 480

points, index_ray, index_tri = mesh.ray.intersects_location(
     origins, vectors, multiple_hits=False)

# Render The mesh for visualization

'''
Depth Information Extraction
'''

# for each hit, find the distance along its vector
depth = trimesh.util.diagonal_dot(points - origins[0],vectors[index_ray])
# find pixel locations of actual hits
pixel_ray = pixels[index_ray]

# create a numpy array we can turn into an image
# doing it with uint8 creates an `L` mode greyscale image
a = np.zeros(scene.camera.resolution, dtype=np.uint8).T

# scale depth against range (0.0 - 1.0)
depth_float = ((depth - depth.min()) / depth.ptp())

# convert depth into 0 - 255 uint8
depth_int = (depth_float * 255).round().astype(np.uint8)
# assign depth to correct pixel locations
#a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
a[pixel_ray[:, 1], pixel_ray[:, 0]] = depth_int
a = np.flip(a, axis = 0)
a = np.flip(a, axis = 1)
# create a PIL image from the depth queries

img = PIL.Image.fromarray(a)


# show the resulting image
if visualize_depth:
  img.show()

# create a raster render of the same scene using OpenGL
# rendered = PIL.Image.open(trimesh.util.wrap_as_stream(scene.save_image()))


# Visulization
if visualize_3d:
  face_colors = np.ones_like(mesh.faces)*192
  face_colors[index_tri] = np.array([255, 0, 0])

  mesh.visual.face_colors = face_colors

  scene = mesh.scene()
  camera_marker = trimesh.creation.camera_marker(camera, marker_height=0.4, origin_size=None)
  scene.add_geometry(camera_marker)
  scene.show()






