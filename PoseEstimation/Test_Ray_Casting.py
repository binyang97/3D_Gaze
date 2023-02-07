from sys import platform
import os
import cv2
import json
from pupil_apriltags import Detector
import open3d as o3d
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from Apriltag_Colmap import create_geometry_at_points, visualize_2d
import copy

if platform == "linux" or platform == "linux2":  
    # linux
        data_path  = "/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/Test2"
elif platform == "win32":
# Windows...
    data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1-dataset2"

images_path = os.path.join(data_path, "frames")
pose_path = os.path.join(data_path, "pose")
mesh_fullpath = os.path.join(data_path, "data3d/textured_output.obj")
depth_path = os.path.join(data_path, "depth")

at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )


images_files = os.listdir(images_path)

images_files.sort()

index = 82

image_file = images_files[index]

camera_param_file = image_file.replace(".jpg", ".json")

img = cv2.imread(os.path.join(images_path, image_file), cv2.IMREAD_GRAYSCALE)
#depth = cv2.imread(os.path.join(depth_path, depth_file))
#print(np.unique(depth))

img_height, img_width = img.shape

tags = at_detector.detect(img)
tag = tags[0]

with open(os.path.join(pose_path, camera_param_file), 'r') as f:
            camera_param = json.load(f)

intrinsics = np.array(camera_param["intrinsics"]).reshape(3, 3)

projectionMatrix = np.array(camera_param["projectionMatrix"]).reshape(4, 4)


Cam2World = np.array(camera_param["cameraPoseARFrame"]).reshape(4, 4)

R_world2cam = Cam2World[:3, :3].T
t_world2cam = - R_world2cam @ Cam2World[:3, 3].reshape(3, 1)
World2Cam = np.concatenate(
                    [np.concatenate([R_world2cam, t_world2cam], axis=1), np.array([[0, 0, 0, 1]])], axis=0)


tag_center = np.array(tag.center).reshape(2,1)

#print(tag_center)
#tag_center[0] = (tag_center[0]-intrinsics[0, 2])/img_width
#tag_center[1] = (tag_center[1]-intrinsics[1, 2])/img_height
#print(tag_center)

uv = np.concatenate((tag_center, np.ones((1,1))), axis = 0)

K_inv = np.linalg.inv(intrinsics)

project_point_cam = K_inv @ uv
project_point_cam = np.concatenate((project_point_cam, np.ones((1,1))), axis = 0)
project_point_world = Cam2World @ project_point_cam

project_point_world = project_point_world[:3]
camera_origin_world = Cam2World[:3, 3].reshape(3, 1)

direction = project_point_world - camera_origin_world

direction_normalized = -direction / np.linalg.norm(direction)


print(direction_normalized)




mesh =  o3d.io.read_triangle_mesh(mesh_fullpath, True)
mesh_in_scene = copy.deepcopy(mesh)
mesh_in_scene = o3d.t.geometry.TriangleMesh.from_legacy(mesh_in_scene)
# Create scene and add the cube mesh
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_in_scene)

ray_origin_direction = np.concatenate((camera_origin_world.reshape(-1), direction_normalized.reshape(-1)), axis=0)



rays = o3d.core.Tensor([ray_origin_direction],
                       dtype=o3d.core.Dtype.Float32)

ans = scene.cast_rays(rays)



depth = ans['t_hit'].numpy()[-1] # z-axis points to the

target_point_3d = camera_origin_world + direction_normalized*depth

VISUALIZATION = True

if VISUALIZATION:
        # mesh = o3d.io.read_triangle_mesh(mesh_fullpath, True)
        #
        # mesh.transform(Cam2World)
        
        coordinate = mesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        

        tag_points = create_geometry_at_points([target_point_3d, camera_origin_world, project_point_world], color = [1, 0, 0], radius=0.05)


        o3d.visualization.draw_geometries([mesh, coordinate, tag_points])


##### Ray Casting for whole image, we could replace it for my previous implementation with trimesh (open 3d is faster)
# mesh =  o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_fullpath, True))

# # Create scene and add the cube mesh
# scene = o3d.t.geometry.RaycastingScene()
# scene.add_triangles(mesh)

# # Rays are 6D vectors with origin and ray direction.
# # Here we use a helper function to create rays for a pinhole camera.
# rays = scene.create_rays_pinhole(intrinsic_matrix = intrinsics,
#                                 extrinsic_matrix = World2Cam,
#                                  width_px=int(tag.center[0]),
#                                  height_px=int(tag.center[1]))

# # Compute the ray intersections.
# ans = scene.cast_rays(rays)

# print(ans['t_hit'])