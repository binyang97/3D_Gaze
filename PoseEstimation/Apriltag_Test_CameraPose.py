import numpy as np
from pupil_apriltags import Detector
from sys import platform
from glob import glob
import cv2
import collections

TagPose = collections.namedtuple(
    "Pose", ["tag_id", "R", "t", "error"])

if __name__ == "__main__":
    
    if platform == "linux" or platform == "linux2":  
    # linux
        images_path = r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/room1/images_gt_apriltags_undistorted"

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



    tags_in_images = []
    tags_id_in_images = []

    for image_fullpath in image_list:
    
        img = cv2.imread(image_fullpath, cv2.IMREAD_GRAYSCALE)
        tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = camera_params, tag_size=0.02)
        tags_in_image = []
        tags_id_in_image = []
        for tag in tags:
            tags_in_image.append(TagPose(tag_id=tag.tag_id, 
                                        R = tag.pose_R, 
                                        t = tag.pose_t, error = tag.pose_err))
            
            tags_id_in_image.append(tag.tag_id)
        tags_in_images.append(tags_in_image)
        tags_id_in_images.append(tags_id_in_image)

    print(tags_id_in_images)
    Match = [] # store the tuple (a,b), a: the index of matched image, b: corresponding tag id
    

    