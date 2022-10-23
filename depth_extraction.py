from operator import concat
import os
from tkinter import W
from xml.dom import NotFoundErr 
import numpy as np
import trimesh
import json

import cv2
import matplotlib.pyplot as plt
import glob

def concatenate_all_points(points_in_room):
    all_points = []
    for points_of_roompart in points_in_room:
        all_points.extend(list(points_of_roompart))
    return np.array(all_points)
 


if __name__ == "__main__":

    id = 1

    bgr_paths = glob.glob('/home/biyang/Documents/3D_Gaze/test_dataset/rgb/*')
    #depth_paths = glob.glob('/home/biyang/Documents/3D_Gaze/test_dataset/depth/*')
    json_paths = glob.glob('/home/biyang/Documents/3D_Gaze/test_dataset/pose/*')

    print(bgr_paths[1], json_paths[0])

    bgr = cv2.imread(bgr_paths[id])
    #depth = cv2.imread(depth_paths[id], cv2.IMREAD_UNCHANGED)

    point_cloud = np.load('/home/biyang/Documents/3D_Gaze/dataset/point_cloud.npz', allow_pickle=True)

    with open(json_paths[0], 'r') as f:
        camera_params = json.load(f)
    room_names = list(point_cloud.keys())

    substring_length = 0
    target_room_name = 0
    for room_name in room_names:
        if room_name in  camera_params['room']:
            if len(room_name) > substring_length:
                target_room_name = room_name
    try: 
        points = point_cloud[target_room_name][1]
    except:
        raise NotFoundErr


    # # Visulization 
    # points = concatenate_all_points(points)
    # color = np.tile(np.array([255, 0, 0]), (len(points), 1))
    # pc = trimesh.PointCloud(vertices = points, colors = color)
    # pc.show(line_settings={'point_size':2})

    points = concatenate_all_points(points)

    print(points.shape)

    # Camera pose transformation
    world2cam = np.array(camera_params['camera_rt_matrix'])
    world2cam = np.vstack((world2cam, np.array([0,0,0,1])))

    points_homo = np.hstack((points, np.ones((len(points), 1))))

    points_cam_coord = np.dot(world2cam, points_homo.T).T

    points_cam_coord = points_cam_coord[:, :3]

    
    # Filter valid points out 
    EPS = 1.0e-16
    valid = points_cam_coord[:, 2] > EPS
    points_valid = points_cam_coord[valid]

    print(points_valid[5])

    # Camera projection
    K = np.array(camera_params['camera_k_matrix'])
    #uv = points_valid @ K.T
    uv = np.dot(K, points_valid.T).T

    z = uv[:, 2]
    uv = uv[:, :2]

    uv = np.round(uv).astype(int)
    #uv[:, :2] = np.round(uv[:, :2], decimals = 0).astype(int)
    # Remove projected points out of image plane boundry
    
    img_width, img_height, _ = bgr.shape

    valid_2d = np.bitwise_and(np.bitwise_and((uv[:, 0] >= 0), (uv[:, 0] < img_width)),
                            np.bitwise_and((uv[:, 1] >= 0), (uv[:, 1] <img_height)))

    
    uv_valid = uv[valid_2d]
    z_valid = z[valid_2d]

    # Generate depth map

    img_z = np.full((img_height, img_width), np.inf)

    # Far object is blocked by the closer object

    for uv, z in zip(uv_valid, z_valid):
        u = uv[0]
        v = uv[1]
        img_z[v, u] = min(img_z[v, u], z)


    # Removing small hole and transmission effect

    #img_z_shift = np.array([img_z, 
    #                      np.roll(img_z, 1, axis = 0),\
    #                        np.roll(img_z, -1, axis= 0),
    #                         np.roll(img_z, 1, axis = 1),
    #                         np.roll(img_z, -1, axis = 1)])

    #img_z = np.min(img_z_shift, axis = 0)

    #img_z = img_z.T
    #np.random.seed(0)
    num_display_pixels = 50
    non_inf_pixels = np.where(img_z != np.inf)


    pixels = np.vstack((non_inf_pixels[0], non_inf_pixels[1])).T

    indices = np.arange(len(pixels))


    
    choices = np.random.choice(indices, size = num_display_pixels)
    chosen_pixels = pixels[choices]

    #print(chosen_pixels)
    #print(img_z[87, 579])

    #img_z[img_z == np.inf] = 255
    
    
    #print(1080*1080-len(img_z[img_z == np.inf]))

    #depth[depth == 255] = 0


    rgb = bgr[...,::-1]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(chosen_pixels[:, 1], chosen_pixels[:, 0], 'o')
    for pixel in chosen_pixels:
        
        ax.text(pixel[1], pixel[0], str(np.round(img_z[pixel[0], pixel[1]], decimals=1)), style='italic', fontsize = 10, color = 'red')



    plt.imshow(rgb)
    plt.show()



    #plt.imshow(img_z,cmap='gray')

    #plt.show()



    #R = world2cam[:3, :3]
    #t = world2cam[:3, 3]
    #R_new = R.T
    #t_new = np.dot(-R_new, t)

    #print(t_new)


    



    

    

    