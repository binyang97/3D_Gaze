from pupil_apriltags import Detector, Detection
from sys import platform
from glob import glob
import cv2
import os
from Colmap_Reader import ColmapReader
import numpy as np
import collections

Mask = collections.namedtuple(
    "Mask", ["tag_id", "tag_corners", "tag_center", "pixels_inside_tag", "mask"])

Points3D = collections.namedtuple(
    "Points3D", ["tag_id", "image_name", "points_3d_xyz"])

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


def get_corresponding_3d_points(masks, keypoints_2d, keypoints_3d_ids):  
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

    database_colmap = ColmapReader(database_path)
    cameras, images, points3D = database_colmap.read_sparse_model()

    # Test
    test_frame = images[60]
    print(np.array(test_frame.xys))
    print(np.where(test_frame.point3D_ids != -1)[0].shape)
    print(test_frame.point3D_ids.shape)

    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    img_fullpath = os.path.join(dataset_path, test_frame.name)
    img = cv2.imread(img_fullpath, cv2.IMREAD_GRAYSCALE)
    tags = at_detector.detect(img)
    
    masks = get_mask(img_grayscale=img, tags=tags, visualization=VISUALIZATION)

    # Find the corresponding 3d points
    search_space_pixels = np.array(test_frame.xys.astype(int))
    points_3d = test_frame.point3D_ids
    
    corresponding_3d_ids, found_2d_pixels = get_corresponding_3d_points(masks, search_space_pixels, points_3d)

    #print(len(found_2d_pixels[2]), len(corresponding_3d_ids[2]))

    apriltag_3d_points = []
    # please use the namedtuple for 3d points as well (so we could also track the corresponding 2d image backward)
    valid_points_3D = []
    for (mask, points_3d_ids, pixels_2d) in zip(masks, corresponding_3d_ids, found_2d_pixels):
        xyz_3d = []
        for i, id_3d in enumerate(points_3d_ids):
            if id_3d != -1:
                xyz_3d.append(points3D[id_3d])
        
        valid_points_3D.append(Points3D(tag_id=mask.tag_id,
                                        image_name=test_frame.name, 
                                        points_3d_xyz=xyz_3d))



    print(valid_points_3D)
        



    if SIMPLE_VISUALIZATION:
        image_list = glob(dataset_path + "/*.jpg")
        image_list.sort()
        index = 80
        
        img = cv2.imread(image_list[index], cv2.IMREAD_GRAYSCALE)

        tags = at_detector.detect(img)

        visualize(img, tags)
        

        