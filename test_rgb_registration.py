import trimesh
import cv2
import mesh_raycast

import numpy as np

import matplotlib.pyplot as plt

import PIL

import time

## Load the mesh
mesh = trimesh.load_mesh("/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0.ply")

#print(mesh.faces.shape)


# Select a rgb image as the input

rgb = cv2.imread("/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0/color_00.jpg")

# Camera Paramters

# Extrinsic Parameters (world to camera)

# T = np.array([[-0.00769736], [-0.00856763], [-0.000635727]])
# R = np.array([[0.999914, -0.00987007, 0.00863509],
#                 [0.0099927, 0.999848, -0.0142716],
#               [-0.00849289, 0.0143568, 0.999861]])


T = np.array([[0.3019], [0.177447], [-0.055902]])
R = np.array([[-0.836422, 0.427017, -0.343591],
                [-0.214059, 0.322595, 0.922015],
              [0.504557, 0.844742 , -0.178419]])
RT = np.concatenate(((R, T)), axis=-1)
world2cam = np.vstack((RT, np.array([0, 0, 0, 1])))

# Intrinsic Paramters

K = np.array([[582.871, 0, 320],
            [0, 582.871, 240],
            [0, 0, 1]])

# Apply the transform to camera coordinate
mesh.apply_transform(world2cam)

scene = mesh.scene()

# any of the automatically generated values can be overridden

# set resolution, in pixels
scene.camera.resolution = [640, 480]
# set field of view, in degrees
# make it relative to resolution so pixels per degree is same
# scene.camera.fov = 60 * (scene.camera.resolution /
#                             scene.camera.resolution.max())

#scene.camera.focal = [1383.45, 1382.48]

scene.camera.K = K

camera = scene.camera
# convert the camera to rays with one ray per pixel
origins, vectors, pixels = scene.camera_rays()

#origins = np.zeros_like(origins)

# do the actual ray- mesh queries (trimesh with pyembree)
# The test time is about 1 second per image with resolution 640 x 480

points, index_ray, index_tri = mesh.ray.intersects_location(
     origins, vectors, multiple_hits=False)

# Render The mesh for visualization


#print(origins[:3], origins.shape)


face_colors = np.ones_like(mesh.faces)*192
face_colors[index_tri] = np.array([255, 0, 0])

mesh.visual.face_colors = face_colors

scene = mesh.scene()
#sphere = trimesh.creation.icosphere(subdivisions=3, radius= 10, color=[0, 255, 0])
camera_marker = trimesh.creation.camera_marker(camera, marker_height=0.4, origin_size=None)
scene.add_geometry(camera_marker)
scene.show()



# mesh_raycast

#result = mesh_raycast.raycast(source=origins, direction=vectors, mesh=mesh)


'''
Depth Information Extraction
'''

# # for each hit, find the distance along its vector
# depth = trimesh.util.diagonal_dot(points - origins[0],vectors[index_ray])
# # find pixel locations of actual hits
# pixel_ray = pixels[index_ray]

# # create a numpy array we can turn into an image
# # doing it with uint8 creates an `L` mode greyscale image
# a = np.zeros(scene.camera.resolution, dtype=np.uint8)

# # scale depth against range (0.0 - 1.0)
# depth_float = ((depth - depth.min()) / depth.ptp())

# # convert depth into 0 - 255 uint8
# depth_int = (depth_float * 255).round().astype(np.uint8)
# # assign depth to correct pixel locations
# a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
# # create a PIL image from the depth queries
# img = PIL.Image.fromarray(a)

# # show the resulting image
# img.show()

# create a raster render of the same scene using OpenGL
# rendered = PIL.Image.open(trimesh.util.wrap_as_stream(scene.save_image()))












