import numpy as np
from sys import platform
from glob import glob
from load_ARSceneData import LoadARSceneData
import cv2
import open3d as o3d
import trimesh
import re
import os
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as R
import copy
#Rendering image for the visualization
import pyrender
import trimesh
import matplotlib.pyplot as plt
from PIL import Image


mesh_trimesh = trimesh.load(r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1-dataset2\data3d\textured_output.obj")
im = Image.open(r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1-dataset2\data3d\textured_output.jpg")
#tex = trimesh.visual.TextureVisuals(image=im)
#print(mesh_trimesh.visual.uv)
#mesh_trimesh.visual.texture = tex
color = trimesh.visual.uv_to_color(mesh_trimesh.visual.uv, im)
mesh_trimesh.visual.color = color

camera_pose = np.array([[-0.91805858, -0.3806811 ,  0.11068129,  0.21480709],
       [ 0.35735638, -0.9155172 , -0.18472863, -0.50719932],
       [ 0.17165333, -0.13003904,  0.97653724, -3.05254395],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
# pyrender.Viewer(scene, use_raymond_lighting=True)


camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2


PI_intrinsics = np.array([[766.2927454396544, 0.0, 543.6272327745995],
                                [0.0, 766.3976103393867, 566.0580149497666],
                                [0.0, 0.0, 1.0]])

img_width = 1088
img_height = 1080

scene.add(camera, pose=camera_pose)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(img_width, img_height)
color, depth = r.render(scene)
# plt.figure()
# plt.subplot(1,2,1)
# plt.axis('off')
plt.imshow(color)
plt.axis('off')
plt.show()
# plt.subplot(1,2,2)
# plt.axis('off')
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

# mesh_gt = o3d.io.read_triangle_mesh(r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1-dataset2\data3d\textured_output.obj", True)
# mesh = copy.deepcopy(mesh_gt)
# world2cam = np.linalg.inv(np.array([[-0.91805858, -0.3806811 ,  0.11068129,  0.21480709],
#        [ 0.35735638, -0.9155172 , -0.18472863, -0.50719932],
#        [ 0.17165333, -0.13003904,  0.97653724, -3.05254395],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]))
# mesh.transform(world2cam)


# PI_intrinsics = np.array([[766.2927454396544, 0.0, 543.6272327745995],
#                                 [0.0, 766.3976103393867, 566.0580149497666],
#                                 [0.0, 0.0, 1.0]])

# img_width = 1088
# img_height = 1080

# render = rendering.OffscreenRenderer(img_width, img_height)

# # setup camera intrinsic values
# pinhole = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, PI_intrinsics[0, 0], PI_intrinsics[1, 1], PI_intrinsics[0, 2], PI_intrinsics[1, 2])
    
# # Pick a background colour of the rendered image, I set it as black (default is light gray)
# render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

# mtl = o3d.visualization.rendering.MaterialRecord()
# mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
# mtl.shader = "defaultUnlit"

# # add mesh to the scene
# render.scene.add_geometry("MyMeshModel", mesh, mtl)
# #render.scene.add_geometry(mesh)

# # render the scene with respect to the camera
# render.scene.camera.set_projection(PI_intrinsics, 0.1, 1.0, img_width, img_height)
# img_o3d = render.render_to_image()

# o3d.visualization.draw_geometries([img_o3d])



# img_rendered = np.asarray(img_o3d)
# img_rgb = img_rendered[...,::-1].copy()

