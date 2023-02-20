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
from create_rgb_from_pcd import render_image
import copy
import matplotlib.pyplot as plt
from imageio import imread, imwrite

def to_arr(m):
    m = np.array(m)
    shape = int(np.sqrt(m.shape[0]))
    return m.reshape((shape,shape))

def project_point_mvp(p_in, mvp, image_width, image_height):
    p0 = np.append(p_in, [1])
    e0 = np.dot(mvp, p0)
    e0[:3] /= e0[3]
    pos_x = e0[0]
    pos_y = e0[1]
    px = (0.5 + (pos_x) * 0.5) * image_width
    py = (1.0 - (0.5 + (pos_y) * 0.5)) * image_height
    return px, py

def Unproject(points, Z, intrinsic):
  f_x = intrinsic[0, 0]
  f_y = intrinsic[1, 1]
  c_x = intrinsic[0, 2]
  c_y = intrinsic[1, 2]
  # This was an error before
  # c_x = intrinsic[0, 3]
  # c_y = intrinsic[1, 3]

  # Step 1. Undistort.
#   points_undistorted = np.array([])
#   if len(points) > 0:
#     points_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=1), intrinsic, distortion, P=intrinsic)
#   points_undistorted = np.squeeze(points_undistorted, axis=1)

  # Step 2. Reproject.
  result = []
  for idx in range(len(points)):
    z = Z[0] if len(Z) == 1 else Z[idx]
    x = (points[idx][0] - c_x) / f_x * z
    y = (points[idx][1] - c_y) / f_y * z
    result.append([x, y, z])
  return result

if platform == "linux" or platform == "linux2":  
    # linux
    data_path  = "/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/Apriltag2-d3"
elif platform == "win32":
# Windows...
    data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag2-d3"

images_path = os.path.join(data_path, "frames")
pose_path = os.path.join(data_path, "pose")
mesh_fullpath = os.path.join(data_path, "data3d/textured_output.obj")
depth_path = os.path.join(data_path, "depth")
conf_path = os.path.join(data_path, "conf")
info_fullpath = os.path.join(data_path, "data3d/info.json")

with open(info_fullpath, "r") as f:
    info = json.load(f)

MW = info["transformToWorldMap"]
user_obb = info["userOBB"] # corner points of oriented bounding box

points_user_obb = user_obb["points"]

MatrixToWorldMap = np.array([[MW["m11"], MW["m12"], MW["m13"], MW["m14"]],
                    [MW["m21"], MW["m22"], MW["m23"], MW["m24"]],
                    [MW["m31"], MW["m32"], MW["m33"], MW["m34"]],
                    [MW["m41"], MW["m42"], MW["m43"], MW["m44"]]]).T

# mesh =  o3d.io.read_triangle_mesh(mesh_fullpath, True)
# mesh.transform(MatrixToWorldMap)
# coordinate = mesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
# o3d.visualization.draw_geometries([mesh, coordinate])

# exit()

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

#index = 82
#index = 113
index = 39
#index = 219

image_file = images_files[index]

camera_param_file = image_file.replace(".jpg", ".json")
depth_file = image_file.replace("frame", "depth")
depth_file = depth_file.replace("jpg", "png")
conf_file = depth_file.replace("depth", "conf")
img = cv2.imread(os.path.join(images_path, image_file), cv2.IMREAD_GRAYSCALE)
depth_true = cv2.imread(os.path.join(depth_path, depth_file), cv2.IMREAD_GRAYSCALE)
conf = cv2.imread(os.path.join(conf_path, conf_file), cv2.IMREAD_GRAYSCALE)

img_height, img_width = img.shape
with open(os.path.join(pose_path, camera_param_file), 'r') as f:
            camera_param = json.load(f)

intrinsics = np.array(camera_param["intrinsics"]).reshape(3, 3)
projectionMatrix = to_arr(camera_param["projectionMatrix"])
Cam2World = np.array(camera_param["cameraPoseARFrame"]).reshape(4, 4)

fxfycxcy= [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
    
# The real size of the tag is about 8.7 cm
tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=0.087)

tag = tags[0]

# Test 3                                                                                                                                                    
# distance_tag2cam = np.linalg.norm(np.array(tag.pose_t))


# project_3d_point = Unproject([tag.center], [distance_tag2cam], intrinsics)
# print(project_3d_point)

# r = R.from_euler('xyz', [0, 180, -90], degrees=True)
# Additional_Rotation = r.as_matrix()
# additional_rotation = np.concatenate(
#                     [np.concatenate([Additional_Rotation, np.array([[0], [0], [0]])], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

# project_3d_point = np.array(project_3d_point)[-1].reshape(3, 1)
# project_point_cam = np.concatenate((project_3d_point, np.ones((1,1))), axis = 0)
# project_point_world = (Cam2World @additional_rotation) @ project_point_cam
# target_point_3d = project_point_world[:3]

# target_point_3d_cam = project_3d_point
# target_point_3d_cam = -target_point_3d_cam

# mesh =  o3d.io.read_triangle_mesh(mesh_fullpath, True)
# mesh.transform(np.linalg.inv(Cam2World))
# tag_points = create_geometry_at_points([target_point_3d_cam], color = [1, 0, 0], radius=0.05)
#         #tag_points = create_geometry_at_points([target_point_3d, np.zeros((3,1)), project_point_cam[:3]], color = [1, 0, 0], radius=0.1)


# o3d.visualization.draw_geometries([mesh, tag_points])

# exit()


R_world2cam = Cam2World[:3, :3].T
t_world2cam = - R_world2cam @ Cam2World[:3, 3].reshape(3, 1)
World2Cam = np.concatenate(
                    [np.concatenate([R_world2cam, t_world2cam], axis=1), np.array([[0, 0, 0, 1]])], axis=0)


tag_center = np.array(tag.center).reshape(2,1)
# tag_center[0] = 2*(tag_center[0] - 0.5 * img_width)
# tag_center[1] = 2*(-tag_center[1] + 0.5 * img_height)

uv = np.concatenate((tag_center, np.ones((1,1))), axis = 0)

K_inv = np.linalg.inv(intrinsics)
                 
project_point_cam = K_inv @ uv
project_point_cam = np.concatenate((project_point_cam, np.ones((1,1))), axis = 0)
project_point_world = Cam2World @ project_point_cam


project_point_world = project_point_world[:3]
camera_origin_world = Cam2World[:3, 3].reshape(3, 1)

direction = project_point_world - camera_origin_world

direction_normalized = -direction / np.linalg.norm(direction)




# x = back_project_2d[0]
# y = back_project_2d[1]
# w = back_project_2d[2]
# new_x = x/w
# new_y = y/w

# # print(uv)

# # print(new_x, new_y)




#Test 2
# mesh_no_color = o3d.io.read_triangle_mesh(mesh_fullpath)
# points = []
# valid_points_3d = []

# r = R.from_euler('xyz', [0, 0, 90], degrees=True)
# Additional_Rotation = r.as_matrix()

# additional_rotation = np.concatenate(
#                 [np.concatenate([Additional_Rotation, np.zeros((3,1))], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
# for p3d in np.array(mesh_no_color.vertices)[::20]:
    
#     mvp = np.dot(projectionMatrix, np.linalg.inv(Cam2World @ additional_rotation))
#     p3d_cam = np.append(p3d, [1])
#     p3d_cam =  np.linalg.inv(Cam2World @ additional_rotation) @ p3d_cam
#     pt2d = project_point_mvp(p3d, mvp, img_width, img_height)
    
#     x,y = pt2d[:2]
    
#     if x >= 0 and x < img_width and y >= 0 and y < img_height and p3d_cam[2] < 0:
#         points.append((x,y))
#         valid_points_3d.append(p3d)
    
# points = np.array(points)
# points_3d = np.array(valid_points_3d)


# plt.figure(figsize=(12,12))

# plt.imshow(imread(os.path.join(images_path, image_file)))

# plt.plot( points[:,0], points[:,1] , '.', color='magenta', alpha=0.5)
# plt.show()

# highlight_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d))
# highlight_points.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([mesh_no_color, highlight_points])

# exit()



mesh =  o3d.io.read_triangle_mesh(mesh_fullpath, True)

# r = R.from_euler('xyz', [0, 180, -90], degrees=True)
# Additional_Rotation = r.as_matrix()
# additional_rotation = np.concatenate(
#                     [np.concatenate([Additional_Rotation, np.array([[0], [0], [0]])], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

# mesh.transform(additional_rotation)

mesh_in_scene = copy.deepcopy(mesh)
mesh_in_scene = o3d.t.geometry.TriangleMesh.from_legacy(mesh_in_scene)
# Create scene and add the cube mesh
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_in_scene)

ray_origin_direction = np.concatenate((camera_origin_world.reshape(-1), direction_normalized.reshape(-1)), axis=0)
#ray_origin_direction = np.concatenate((np.zeros(3), direction_cam_normalized.reshape(-1)), axis=0)
print(ray_origin_direction)


rays = o3d.core.Tensor([ray_origin_direction],
                       dtype=o3d.core.Dtype.Float32)

ans = scene.cast_rays(rays)



depth = ans['t_hit'].numpy()[-1] # z-axis points to the

target_point_3d = camera_origin_world + direction_normalized*depth



# Test
# target_point_3d = target_point_3d + np.array([0.1, 0.1, -1]).reshape(3, 1)
# r = R.from_euler('xyz', [0, 0, 90], degrees=True)
# Additional_Rotation = r.as_matrix()

# additional_rotation = np.concatenate(
#                 [np.concatenate([Additional_Rotation, np.zeros((3,1))], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
# mvp = np.dot(projectionMatrix, np.linalg.inv(Cam2World @ additional_rotation))
# new_x, new_y = project_point_mvp(target_point_3d, mvp, img_width, img_height)
# plt.figure(figsize=(12,12))

# plt.imshow(imread(os.path.join(images_path, image_file)))

# plt.plot( new_x, new_y, '.', color='magenta', alpha=0.5)
# plt.show()



# exit()




VISUALIZATION = False
RENDERING = True

if VISUALIZATION:
        # mesh = o3d.io.read_triangle_mesh(mesh_fullpath, True)
        #
        #mesh.transform(World2Cam)
        
        coordinate = mesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
       


        # coordinate_2 = copy.deepcopy(coordinate)

        # coordinate.transform(Cam2World)
        # coordinate_2.transform(Cam2World @ additional_rotation)
        #user_obb_points_3d = create_geometry_at_points(points_user_obb, color = [1, 0, 0], radius = 0.05)
        #o3d.visualization.draw_geometries([mesh, coordinate, user_obb_points_3d])
        

        tag_points = create_geometry_at_points([target_point_3d, camera_origin_world], color = [1, 0, 0], radius=0.02)
        #tag_points = create_geometry_at_points([target_point_3d, np.zeros((3,1)), project_point_cam[:3]], color = [1, 0, 0], radius=0.1)


        o3d.visualization.draw_geometries([mesh, coordinate, tag_points])


##### Ray Casting for whole image, we could replace it for my previous implementation with trimesh (open 3d is faster)
# mesh =  o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_fullpath, True))

if RENDERING:
    # Create scene and add the cube mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_in_scene)


    r = R.from_euler('xyz', [0, 0, 0], degrees=True)
    Additional_Rotation = r.as_matrix()
    additional_rotation = np.concatenate(
                        [np.concatenate([Additional_Rotation, np.array([[0], [0], [0]])], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    print(additional_rotation)

    reflection_xy = np.eye(4)
    reflection_xy[2, 2] = -1
    extrinsic = np.linalg.inv(Cam2World @ additional_rotation @ reflection_xy)
    # Rays are 6D vectors with origin and ray direction.
    # Here we use a helper function to create rays for a pinhole camera.
    rays = scene.create_rays_pinhole(intrinsic_matrix = intrinsics,
                                    extrinsic_matrix = extrinsic,
                                    width_px=img_width,
                                    height_px=img_height)

    # Compute the ray intersections.

    rays_inverse = rays.numpy()

    #rays_inverse[:, :, -1] = -rays_inverse[:, :, -1]

    rays = o3d.core.Tensor(rays_inverse, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)

    #print(ans['t_hit'])

    titles = ['Original Image','Rendered Depth Image', 'Pre-recorded Depth Image']
                # 'Original Image (grayscale)','Image after removing the noise (grayscale)']
    depth_true = cv2.rotate(depth_true, cv2.ROTATE_90_CLOCKWISE)
    depth_resized = cv2.resize(depth_true, (1440, 1920), interpolation = cv2.INTER_AREA)
    
    
    images = [img, ans['t_hit'].numpy(), depth_resized]
    plt.figure(figsize=(13,5))
    for i in range(len(images)):
        plt.subplot(1,3,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

    # plt.imshow(ans['t_hit'].numpy())
    # plt.show()