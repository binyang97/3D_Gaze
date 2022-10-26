import trimesh
import cv2

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

## Load the mesh
mesh = trimesh.load_mesh("/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0.ply")

#print(mesh.faces.shape)


# Select a rgb image as the input

rgb = cv2.imread("/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0/color_00.jpg")

# Camera Paramters

# Extrinsic Parameters (world to camera)

T = np.array([-0.00769736, -0.00856763, -0.000635727]).T
R = np.array([[0.999914, -0.00987007, 0.00863509],
                [0.0099927, 0.999848, -0.0142716],
              [-0.00849289, 0.0143568, 0.999861]])

# Intrinsic Paramters

K = np.array([[1383.45, 0, 979.023],
            [0, 1382.48, 529.022],
            [0, 0, 1]])


test_point = np.array([100, 100])

scene = mesh.scene()

# any of the automatically generated values can be overridden

# set resolution, in pixels
scene.camera.resolution = [640, 480]
# set field of view, in degrees
# make it relative to resolution so pixels per degree is same
scene.camera.fov = 60 * (scene.camera.resolution /
                            scene.camera.resolution.max())

# convert the camera to rays with one ray per pixel
origins, vectors, pixels = scene.camera_rays()

# do the actual ray- mesh queries
points, index_ray, index_tri = mesh.ray.intersects_location(
    origins, vectors, multiple_hits=False)

# for each hit, find the distance along its vector
depth = trimesh.util.diagonal_dot(points - origins[0],
                                    vectors[index_ray])
# find pixel locations of actual hits
pixel_ray = pixels[index_ray]

# create a numpy array we can turn into an image
# doing it with uint8 creates an `L` mode greyscale image
a = np.zeros(scene.camera.resolution, dtype=np.uint8)

# scale depth against range (0.0 - 1.0)
depth_float = ((depth - depth.min()) / depth.ptp())

# convert depth into 0 - 255 uint8
depth_int = (depth_float * 255).round().astype(np.uint8)
# assign depth to correct pixel locations
a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
# create a PIL image from the depth queries
img = PIL.Image.fromarray(a)

# show the resulting image
img.show()

# create a raster render of the same scene using OpenGL
# rendered = PIL.Image.open(trimesh.util.wrap_as_stream(scene.save_image()))












