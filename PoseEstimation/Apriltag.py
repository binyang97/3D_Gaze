from pupil_apriltags import Detector, Detection
from sys import platform
from glob import glob
import cv2
import os
from Colmap_Reader import ColmapReader
import numpy as np
import collections
import open3d as o3d
import copy

Mask = collections.namedtuple(
    "Mask", ["tag_id", "tag_corners", "tag_center", "pixels_inside_tag", "mask"])

# points_3d_info containing information like xyz, id of image and 2d pixel (with which we can backward track the correspondence)
Points3D = collections.namedtuple(
    "Points3D", ["image_name", "points_3d_info"])


def create_geometry_at_points(points, color, radius=0.005):
    geometries = o3d.geometry.TriangleMesh()
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius) #create a small sphere to represent point
        sphere.translate(point) #translate this sphere to point
        geometries += sphere
    geometries.paint_uniform_color(color)
    return geometries 

def visualize(img_grayscale, tags):
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

def get_corresponding_3d_points(ids, all_3d_keypoints):

    valid_points_3D = []
    for points_3d_ids in ids:
        xyz_3d = []
        for id_3d in points_3d_ids:
            if id_3d != -1:
                xyz_3d.append(all_3d_keypoints[id_3d])
        
        valid_points_3D.append(Points3D(image_name=test_frame.name, 
                                        points_3d_info=xyz_3d))

    return valid_points_3D


if __name__ == '__main__':

    if platform == "linux" or platform == "linux2":  
    # linux
        dataset_path = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100_undistorted_chessboard/images_undistorted_chessboard"
        database_path = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100_undistorted_chessboard/"

    elif platform == "win32":
    # Windows...
        dataset_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image100\images_undistorted_prerecorded"
        database_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image_100_undistorted_prerecorded\Stereo_Fusion.min_num_pixels=10"


    SIMPLE_VISUALIZATION = False
    VISUALIZATION = False
    VISUALIZATION_3D = False


    database_colmap = ColmapReader(database_path)
    cameras, images, points3D = database_colmap.read_sparse_model()
    pcd_rc, _ = database_colmap.read_dense_model()

    # Test
    test_frame = images[29]
    # print(np.array(test_frame.xys))
    # print(np.where(test_frame.point3D_ids != -1)[0].shape)
    # print(test_frame.point3D_ids.shape)

    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    for frame in images:

        img_fullpath = os.path.join(dataset_path, test_frame.name)
        img = cv2.imread(img_fullpath, cv2.IMREAD_GRAYSCALE)
        tags = at_detector.detect(img)
        if len(tags) == 0:
            continue

    
        masks = get_mask(img_grayscale=img, tags=tags, visualization=VISUALIZATION)

        # Find the corresponding 3d points
        search_space_pixels = np.array(test_frame.xys.astype(int))
        points_3d = test_frame.point3D_ids
        
        corresponding_3d_ids, _ = get_corresponding_3d_points_ids(masks, search_space_pixels, points_3d)

        #print(len(found_2d_pixels[2]), len(corresponding_3d_ids[2]))

        # please use the namedtuple for 3d points as well (so we could also track the corresponding 2d image backward)
        valid_points_3D = get_corresponding_3d_points(corresponding_3d_ids, points3D)

        
    # define a color bar for visualization
    colorbar = np.array([[0, 0, 255], # blue
                        [255, 0, 0], # red
                        [0, 255, 0], # green
                        [255,140,0], # orange
                        [138,43,226], #purple
                        [255,215,0],] #yellow 
                        ) 
    
    if VISUALIZATION_3D:


        pcd = copy.deepcopy(pcd_rc)
        #pcd.paint_uniform_color([220,220,220])
        vis = [pcd]
        for i, tag in enumerate(valid_points_3D):
            points = []
            for valid_point_3D in tag.points_3d_info:
                points.append(valid_point_3D.xyz)
            
            points = np.array(points)
            

            # highlight the points
            vis_geometry = create_geometry_at_points(points, colorbar[i], radius=0.5)

            vis.append(vis_geometry)

        o3d.visualization.draw_geometries(vis)
         

            



                


    if SIMPLE_VISUALIZATION:
        image_list = glob(dataset_path + "/*.jpg")
        image_list.sort()
        index = 80
        
        img = cv2.imread(image_list[index], cv2.IMREAD_GRAYSCALE)

        tags = at_detector.detect(img)

        visualize(img, tags)
        

        