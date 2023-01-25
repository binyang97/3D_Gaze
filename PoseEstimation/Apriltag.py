from pupil_apriltags import Detector, Detection
from sys import platform
from glob import glob
import cv2
import os
from Colmap_Reader import ColmapReader
import numpy as np
import collections

BaseMask = collections.namedtuple(
    "Mask", ["tag_id", "tag_corners", "tag_center", "pixels_inside_tag"])

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
        mask = np.zeros((img.shape), dtype=np.uint8)
        pts = np.array(tag.corners.astype(int))
        cv2.fillPoly(mask, [pts], (255))

    return
        

if __name__ == '__main__':

    if platform == "linux" or platform == "linux2":  
    # linux
        dataset_path = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100_undistorted_chessboard/images_undistorted_chessboard"
        database_path = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100_undistorted_chessboard/"

    elif platform == "win32":
    # Windows...
        dataset_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image100\images_undistorted_prerecorded"
        database_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image_100_undistorted_prerecorded\Stereo_Fusion.min_num_pixels=10"


    VISUALIZATION = False

    database_colmap = ColmapReader(database_path)
    cameras, images, points3D = database_colmap.read_sparse_model()

    # Test
    test_frame = images[100]
    print(np.array(test_frame.xys).shape)
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

    # print(tags[0].corners)
    # print(tags[0].center)
    mask = np.zeros((img.shape), dtype=np.uint8)
    pts = np.array(tags[0].corners.astype(int))
    print(pts)
    cv2.fillPoly(mask, [pts], (255))

    pixels = np.where(mask == 255)
    print(pixels)
    # cv2.imshow('mask', mask)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if VISUALIZATION:
        image_list = glob(dataset_path + "/*.jpg")
        image_list.sort()
        index = 80
        
        img = cv2.imread(image_list[index], cv2.IMREAD_GRAYSCALE)

        tags = at_detector.detect(img)

        visualize(img, tags)
        

        