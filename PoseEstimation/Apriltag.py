from pupil_apriltags import Detector, Detection
from sys import platform
from glob import glob
import cv2
import os
from Colmap_Reader import ColmapReader
from sklearn.cluster import DBSCAN
import numpy as np
import collections
import open3d as o3d
import copy
import json

Mask = collections.namedtuple(
    "Mask", ["tag_id", "tag_corners", "tag_center", "pixels_inside_tag", "mask"])

# points_3d_info containing information like xyz, id of image and 2d pixel (with which we can backward track the correspondence)
Points3D = collections.namedtuple(
    "Points3D", ["image_name", "points_3d_info"])

# define a color bar for visualizatio
global colorbar 
colorbar = np.array([[0, 0, 255], # blue
                        [255, 0, 0], # red
                        [0, 255, 0], # green
                        [255,20,147], # pink
                        [138,43,226], #purple
                        [255,215,0], # yellow
                        [255,140,0],] # yellow 
                        ) /255

# helper function
def filter_array(arr1, arr2):
    return filter(lambda x: x not in arr2, arr1)

def create_geometry_at_points(points, color, radius=0.005):
    geometries = o3d.geometry.TriangleMesh()
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius) #create a small sphere to represent point
        sphere.translate(point) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color(color)
    return geometries 

def create_pcd(points, color):
    points_open3d = o3d.utility.Vector3dVector(points)
    pcd = o3d.geometry.PointCloud(points_open3d)
    pcd.paint_uniform_color(color)

    return pcd

def visualize_2d(img_grayscale, tags):
    color_img = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2RGB)
    for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(
                    color_img,
                    tuple(tag.corners[idx - 1, :].astype(int)),
                    tuple(tag.corners[idx, :].astype(int)),
                    (0, 255, 0),
                )

            cv2.putText(
                color_img,
                str(tag.tag_id),
                org=(
                    tag.corners[0, 0].astype(int) + 10,
                    tag.corners[0, 1].astype(int) + 10,
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
            )

    cv2.imshow("Detected tags", color_img)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualization_3d(pcd, registered_points, highlight = False):
    vis = [pcd]
    for i, tag_id in enumerate(registered_points.keys()):
        if highlight:
        # highlight the points
            vis_geometry = create_geometry_at_points(registered_points[tag_id], colorbar[i], radius=0.5)
            vis.append(vis_geometry)
        else:
            
            pcd_tag = create_pcd(registered_points[tag_id], colorbar[i])
            vis.append(pcd_tag)

    o3d.visualization.draw_geometries(vis)


def get_mask(img_grayscale, tags, visualization = True):
    print("There are totally {} tags in the frame".format(len(tags)))
    masks = []
    for tag in tags:
        mask = np.zeros((img_grayscale.shape), dtype=np.uint8)
        pts = np.array(tag.corners.astype(int))
        cv2.fillPoly(mask, [pts], (255))

        valid_indices = np.where(mask == 255)
        # in form of u, v
        valid_pixels = np.vstack([valid_indices[1], valid_indices[0]]).T

        masks.append(Mask(tag_id=tag.tag_id,
                        tag_corners=tag.corners,
                        tag_center=tag.center,
                        pixels_inside_tag= valid_pixels,
                        mask=mask))
    if visualization:
        mask_vis = masks[0].mask
        for idx in range(1, len(masks)):
           mask_vis = cv2.bitwise_or(mask_vis, masks[idx].mask, mask = None)

        vis = np.concatenate((img_grayscale, mask_vis), axis = 1)

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)

        cv2.imshow('output', vis)

        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
	
    
    return masks


def get_corresponding_3d_points_ids(masks, keypoints_2d, keypoints_3d_ids):  
    valid_indices_3d_points = []
    valid_indices_2d_pixels = []
    for mask in masks:
        valid_pixels = mask.pixels_inside_tag
        found_indices = []
        found_pixels = []
        for valid_pixel in valid_pixels:
            index = np.where((keypoints_2d == valid_pixel).all(axis=1))[0]
            if len(index) > 0:
                found_indices.extend(list(index))
                found_pixels.extend([valid_pixel] * len(index))
        found_indices = np.array(found_indices)
        valid_indices_3d_points.append(keypoints_3d_ids[found_indices])
        valid_indices_2d_pixels.append(found_pixels)

    # The filtered 3d points still have id of -1 
    return  valid_indices_3d_points, valid_indices_2d_pixels

def get_valid_3d_points(valid_ids, all_3d_keypoints):

    valid_points_3D_xyz = []
    valid_points_3D_repro_error = []
    valid_points_3D_info = []

    for id_3d in valid_ids:
        if id_3d != -1:
            valid_points_3D_info.append(all_3d_keypoints[id_3d])    
            valid_points_3D_xyz.append(all_3d_keypoints[id_3d].xyz)
            valid_points_3D_repro_error.append(all_3d_keypoints[id_3d].error)

    valid_points_3D_xyz = np.array(valid_points_3D_xyz)
    valid_points_3D_repro_error = np.array(valid_points_3D_repro_error)
    return valid_points_3D_xyz, valid_points_3D_repro_error, valid_points_3D_info

def filter_registered_points(registered_points_xyz, registered_points_reprojection_error, visualization = False):
     for key in registered_points_xyz.keys():
        point_cloud = registered_points_xyz[key]
        repro_error = registered_points_reprojection_error[key]

        min_repro_error = np.mean(repro_error) - 0*np.std(repro_error)
        print(min_repro_error)

        valid_repro_error = np.zeros(len(repro_error), dtype=bool)
        valid_repro_error[repro_error < min_repro_error] = True

        outliers_too_large_repro_error = point_cloud[~valid_repro_error]
        point_cloud = point_cloud[valid_repro_error]
        
        core_samples_mask = np.zeros(len(point_cloud), dtype=bool)
        # Define the DBSCAN function
        # eps is the most important parameters
        # minimum distance that will be defined as a neighbor
        dbscan = DBSCAN(eps=0.5, min_samples=5)

        # Fit the model to the point cloud data
        dbscan.fit(point_cloud)
        
        labels = dbscan.labels_
        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        counts = counts[unique_labels >= 0]
        unique_labels = unique_labels[unique_labels >= 0]

        print(unique_labels, counts, key)
        # Case when there is multiple valid clustered classes --> simple idea: directly take the one with the most count
        if len(unique_labels) > 1:
            main_label = unique_labels[np.argmax(counts)]
            core_samples_mask[labels == main_label] = True
        else:
            core_samples_mask[dbscan.core_sample_indices_] = True

        registered_points_xyz[key] = point_cloud[core_samples_mask]

        if outliers is None:
            outliers = point_cloud[~core_samples_mask]
        else:
            outliers = np.concatenate((outliers, point_cloud[~core_samples_mask]), axis=0)

        if outliers is None:
            outliers = outliers_too_large_repro_error
        else:
            outliers = np.concatenate((outliers, outliers_too_large_repro_error), axis=0)

        

        return registered_points_xyz, outliers

if __name__ == '__main__':

    if platform == "linux" or platform == "linux2":  
    # linux
        dataset_path = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100_undistorted_chessboard/images_undistorted_chessboard"
        database_path = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100_undistorted_chessboard/"

    elif platform == "win32":
    # Windows...
        dataset_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image100\images_undistorted_prerecorded"
        database_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image_100_undistorted_prerecorded\Stereo_Fusion.min_num_pixels=10"

    VISUALIZATION_MASK = False
    VISUALIZATION_3D = False
    TEST = False


    database_colmap = ColmapReader(database_path)
    cameras, images, points3D = database_colmap.read_sparse_model()
    pcd_rc, _ = database_colmap.read_dense_model()

    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )


    # test
    if TEST:
        test_frame = images[15]
        img_fullpath = os.path.join(dataset_path, test_frame.name)
        img = cv2.imread(img_fullpath, cv2.IMREAD_GRAYSCALE)
        tags = at_detector.detect(img)
        visualize_2d(img, tags)

        exit()


    registered_3d_points_xyz = {} # register for each tag (including only xyz)
    registered_3d_points_info = {} # register for each tag (including other info)
    registered_3d_points_repro_error = {} # register for each tag (including other info)
    visited_id = []
    for frame in images.values():

        img_fullpath = os.path.join(dataset_path, frame.name)
        img = cv2.imread(img_fullpath, cv2.IMREAD_GRAYSCALE)
        tags = at_detector.detect(img)
        if len(tags) == 0:
            continue

        tag_ids = [tag.tag_id for tag in tags]
        masks = get_mask(img_grayscale=img, tags=tags, visualization=VISUALIZATION_MASK)

        # Find the corresponding 3d points
        search_space_pixels = np.array(frame.xys.astype(int))
        points_3d = frame.point3D_ids
        corresponding_3d_ids, _ = get_corresponding_3d_points_ids(masks, search_space_pixels, points_3d)

        # filter the duplicated ids
        for (tag_id, corresponding_3d_id) in zip(tag_ids, corresponding_3d_ids):
            non_visited_3d_id = filter_array(corresponding_3d_id, visited_id)
            non_visited_3d_id = list(non_visited_3d_id)
            visited_id.extend(non_visited_3d_id)

            corresponding_3d_xyz, corresponding_3d_repro_error, corresponding_3d_info = get_valid_3d_points(non_visited_3d_id, points3D)


            if len(corresponding_3d_xyz) == 0:
                continue
            if tag_id not in registered_3d_points_xyz.keys():
                registered_3d_points_xyz[tag_id] = corresponding_3d_xyz
                registered_3d_points_info[tag_id] = corresponding_3d_info
                registered_3d_points_repro_error[tag_id] = corresponding_3d_repro_error
                
            else:
                registered_3d_points_xyz[tag_id] = np.concatenate((registered_3d_points_xyz[tag_id], corresponding_3d_xyz), axis=0)
                registered_3d_points_info[tag_id] = np.concatenate((registered_3d_points_info[tag_id], corresponding_3d_info), axis=0)
                registered_3d_points_repro_error[tag_id] = np.concatenate((registered_3d_points_repro_error[tag_id], corresponding_3d_repro_error), axis=0)
        
    
    # with open(r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\apriltags_repro_error.json", "w") as outfile:
    #    json.dump({k: v.tolist() for k, v in registered_3d_points_repro_error.items()}, outfile, indent=4)

    filtered_registered_tag_points_xyz, outliers_xyz = filter_registered_points(registered_3d_points_xyz, registered_3d_points_repro_error, visualization = False)

    pcd_copy = copy.deepcopy(pcd_rc)
    pcd_rc.paint_uniform_color(np.array([220, 220 ,220])/255)


    # for tag in registered_3d_points_info.keys():
    #     for info in registered_3d_points_info[tag]:
    #         print(info.error)
    if VISUALIZATION_3D:
        #visualization_3d(pcd_rc, registered_3d_points_xyz, highlight=False)

        pcd_outliers = create_pcd(outliers_xyz, color = [0, 0 ,0])
        vis = [pcd_rc, pcd_outliers]
        for i, tag_id in enumerate(filtered_registered_tag_points_xyz.keys()):
        
            pcd_tag = create_pcd(filtered_registered_tag_points_xyz[tag_id], colorbar[i])
            vis.append(pcd_tag)
        
         

        