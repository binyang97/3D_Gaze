import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sys import platform
import json
from glob import glob
from pathlib import Path
import msgpack
import logging
import pickle


logger = logging.getLogger(__name__)

def _load_object_legacy(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "rb") as fh:
        data = pickle.load(fh, encoding="bytes")
    return data

def load_object(file_path, allow_legacy=True):
    import gc

    file_path = Path(file_path).expanduser()
    with file_path.open("rb") as fh:
        try:
            gc.disable()  # speeds deserialization up.
            data = msgpack.unpack(fh, raw=False)
        except Exception as e:
            if not allow_legacy:
                raise e
            else:
                logger.info(
                    "{} has a deprecated format: Will be updated on save".format(
                        file_path
                    )
                )
                data = _load_object_legacy(file_path)
        finally:
            gc.enable()
    return data



if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":  
    # linux
        pass

    elif platform == "win32":
    # Windows...
        pi_imagepath = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1_v2\images" 
        iphone_imagepath = glob(r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\TestPhone\images" + "/*")
        output_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1_v2\images_undistorted"
        intrinsic_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\raw_data\2023-01-10-23-37-12\world.intrinsics"


    intrinsics = load_object(intrinsic_path)
    print(intrinsics)


    VISUALIZATION = False
    SAVE = False

    id = 20
    pi_intrinsic = {
        "resolution": [1088, 1080],
        "dist_coefs": 
                [
                    -0.12390715699556255,
                    0.09983010007937897,
                    0.0013846287331131738,
                    -0.00036539454816030264,
                    0.020072404577046853,
                    0.2052173022520547,
                    0.009921380887245364,
                    0.06631870205961587,
                ]
            ,
        "camera_matrix": [
                [766.2927454396544, 0.0, 543.6272327745995],
                [0.0, 766.3976103393867, 566.0580149497666],
                [0.0, 0.0, 1.0],
            ],
        "cam_type": "radial",
        }

    mtx = np.array(pi_intrinsic['camera_matrix'])
    dist = np.array(pi_intrinsic['dist_coefs'])

    img_files = os.listdir(pi_imagepath)
    

    if VISUALIZATION:
        img = cv2.imread(os.path.join(pi_imagepath, img_files[id]),1) 
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
        #newcameramtx = mtx
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        titles = ['Original Image(colored)','Undistorted Image']
                # 'Original Image (grayscale)','Image after removing the noise (grayscale)']
        images = [img, dst]
        plt.figure(figsize=(13,5))
        for i in range(len(images)):
            plt.subplot(1,2,i+1)
            plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()


    if SAVE:
        for img_path in img_files:
            img = cv2.imread(os.path.join(pi_imagepath, img_path),1) 
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
            #newcameramtx = mtx
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)


            if os.path.exists(output_path):
                pass
            else:
                os.makedirs(output_path)

            cv2.imwrite(os.path.join(output_path, img_path), dst)


