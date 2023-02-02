import os
from pupil_apriltags import Detector, Detection
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

from Apriltag import visualize_2d


if __name__ == "__main__":

    if platform == "linux" or platform == "linux2":  
        image_path = "/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_phone/Test_Rendering"

    elif platform == "win32":

        image_path = r""



    images = glob(image_path + "/*.jpg")

    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )


    img_fullpath = images[3]
    img = cv2.imread(img_fullpath, cv2.IMREAD_GRAYSCALE)
    tags = at_detector.detect(img)
    print(tags)
    visualize_2d(img, tags)