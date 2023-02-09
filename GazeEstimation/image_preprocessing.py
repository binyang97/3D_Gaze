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

def calibrate(path_chessboard, visualization = True):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob(path_chessboard + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            if visualization:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7,6), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    return ret, mtx, dist, rvecs, tvecs



if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":  
    # linux
        pi_imagepath = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100/images" 
        output_path = r"/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_image100_undistorted_chessboard/images_undistorted_chessboard"
        intrinsic_path = ""
        chessboard_path = r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/chessboard"

    elif platform == "win32":
    # Windows...
        pi_imagepath = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\office1\data_1\images" 
        output_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\office1\data_1\images_undistorted"
        intrinsic_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\raw_data\2023-01-10-23-37-12\world.intrinsics"
        chessboard_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\chessboard"

    VISUALIZATION = False
    SAVE = True
    CHESSBOARD = False

    ## Intrinsics parameters are totally the same

    if os.path.exists(intrinsic_path):
        intrinsics = load_object(intrinsic_path)
        mtx = np.array(intrinsics['(1088, 1080)']['camera_matrix'])
        dist = np.array(intrinsics['(1088, 1080)']['dist_coefs'])

    else:
    
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

    _, mtx_chessboard , dist_chessboard , _,_ = calibrate(chessboard_path, visualization=False)

    
    img_files = os.listdir(pi_imagepath)
    
    id = 10

    if VISUALIZATION:
        #  Undistortion the image
        img = cv2.imread(os.path.join(pi_imagepath, img_files[id]),1) 
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
        #newcameramtx = mtx
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Smoothing (Using Gaussian Filter)
        #dst2 = cv2.GaussianBlur(dst, (7, 7), 0)

        newcameramtx_chessboard , roi_chessboard  = cv2.getOptimalNewCameraMatrix(mtx_chessboard , dist_chessboard , (w,h), 0, (w,h))
        dst_chessboard = cv2.undistort(img, mtx_chessboard , dist_chessboard , None, newcameramtx_chessboard )

        titles = ['Original Image(colored)','Undistorted Image (Pre-recorded)', 'Undistorted Image (Chessboard)']
                # 'Original Image (grayscale)','Image after removing the noise (grayscale)']
        images = [img, dst, dst_chessboard]
        plt.figure(figsize=(13,5))
        for i in range(len(images)):
            plt.subplot(1,3,i+1)
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
            if CHESSBOARD:
                print("The camera intrinsics are from chessboard calibration")
                newcameramtx_chessboard , roi_chessboard  = cv2.getOptimalNewCameraMatrix(mtx_chessboard , dist_chessboard , (w,h), 0, (w,h))
                dst= cv2.undistort(img, mtx_chessboard , dist_chessboard , None, newcameramtx_chessboard )
            else:
                print("The camera intrinsics are pre-recorded")
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
                #newcameramtx = mtx
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            #dst2 = cv2.GaussianBlur(dst, (7, 7), 0)


            if os.path.exists(output_path):
                pass
            else:
                os.makedirs(output_path)

            cv2.imwrite(os.path.join(output_path, img_path), dst)

    
    
    
            

    