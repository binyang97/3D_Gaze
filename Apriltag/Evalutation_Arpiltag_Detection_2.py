from sys import platform
import os
import cv2
import json
from pupil_apriltags import Detector
import open3d as o3d
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import math

if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":  
        # linux
        json_path = "/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/evaluation_apriltag_detection_PI.json"
    elif platform == "win32":
    # Windows...
        json_path = r""


    with open(json_path, "r") as f:
        evaluation  = json.load(f)


    for tag_id in evaluation.keys():


        X = evaluation[tag_id]["Frame_id"]
        X = [id[0:6] for id in X]
        

        fig=plt.figure()
        fig.suptitle("Tag ID: " + tag_id)
        # plt.plot(X, evaluation[tag_id]["Error_Accuracy"], label = "Relative Error of Alpha")
        # plt.plot(X, evaluation[tag_id]["angle_x"], label = "euler angle on x-axis")
        # plt.plot(X, evaluation[tag_id]["angle_y"], label = "euler angle on y-axis")
        # plt.plot(X, evaluation[tag_id]["angle_z"], label = "euler angle on z-axis")
        # plt.plot(X, evaluation[tag_id]["Real_Distance"], label = "Distance between camera origin and tag center")
        # plt.legend()
        # plt.show()
        

        # X = evaluation[tag_id]["Error_Accuracy"]
        #figure, axis = plt.subplots(2, 2)

        ax1 = plt.subplot2grid(shape=(2,8), loc=(0,0), colspan=2)
        ax2 = plt.subplot2grid((2,8), (0,3), colspan=2)
        ax3 = plt.subplot2grid((2,8), (0,6), colspan=2)
        ax4 = plt.subplot2grid((2,8), (1,1), colspan=2)
        ax5 = plt.subplot2grid((2,8), (1,5), colspan=2)

        fig.add_axes(ax1)
        fig.add_axes(ax2)
        fig.add_axes(ax3)
        fig.add_axes(ax4)
        fig.add_axes(ax5)

        distance_error = [alpha_error * 0.087 for alpha_error in evaluation[tag_id]["Error_Accuracy"]]

        ax1.plot(X, distance_error)
        ax1.set_ylabel("Distance Error (alpha_error * real_tag_size)")
        ax1.set_xlabel("Frame ID")
        ax1.xaxis.set_tick_params(labelsize=7, labelrotation = 60)


        ax2.plot(X, evaluation[tag_id]["angle_x"])
        ax2.set_ylabel("euler angle on x-axis")
        ax2.set_xlabel("Frame ID")
        ax2.xaxis.set_tick_params(labelsize=7, labelrotation = 60)

        ax3.plot(X, evaluation[tag_id]["angle_y"])
        ax3.set_ylabel("euler angle on y-axis")
        ax3.set_xlabel("Frame ID")
        ax3.xaxis.set_tick_params(labelsize=7, labelrotation = 60)

        ax4.plot(X, evaluation[tag_id]["angle_z"])
        ax4.set_ylabel("euler angle on z-axis")
        ax4.set_xlabel("Frame ID")
        ax4.xaxis.set_tick_params(labelsize=7, labelrotation = 60)

        ax5.plot(X, evaluation[tag_id]["Real_Distance"])
        ax5.set_ylabel("Distance between camera origin and tag center")
        ax5.set_xlabel("Frame ID")
        ax5.xaxis.set_tick_params(labelsize=7, labelrotation = 60)

        plt.show()

