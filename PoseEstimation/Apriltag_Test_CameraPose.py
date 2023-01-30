import numpy as np
from pupil_apriltags import Detector
from sys import platform
from glob import glob
import cv2

if __name__ == "__main__":
    
    if platform == "linux" or platform == "linux2":  
    # linux
        images_path = r""

    elif platform == "win32":
    # Windows...
        images_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\images_gt_apriltags_undistorted"
        
    camera_params = [766.2927454396544, 766.3976103393867, 543.6272327745995, 566.0580149497666]
    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )


    image_list = glob(images_path+ "/*.jpg")


    for image_fullpath in image_list:
    
        img = cv2.imread(image_fullpath, cv2.IMREAD_GRAYSCALE)
        tags = at_detector.detect(img, estimate_tag_pose=True,camera_params = camera_params, tag_size=0.02)


    print(tags[0].pose_t)
    print(tags[0].pose_R)
    print(tags[0].pose_err)
    