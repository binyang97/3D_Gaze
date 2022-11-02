
import trimesh
import cv2
#import mesh_raycast

import trimesh_new
from functions import pySaliencyMap

import numpy as np

import matplotlib.pyplot as plt

import PIL

import time

import glob

import os

import pyembree
## Load the data
#mesh = trimesh.load_mesh("/home/biyang/Documents/3D_Gaze/dataset/apt0/apt0.ply")

current_path = os.getcwd()

files = glob.glob(current_path + "/dataset/apt0/apt0/*")

index = 11

cam_files = []
rgb_files = []

for file in files:
    if file.__contains__('.cam'):
        cam_files.append(file)
    elif file.__contains__('color'):
        rgb_files.append(file)

cam_files.sort()
rgb_files.sort()

img = cv2.imread(rgb_files[index])
img_height, img_width, _ = img.shape

sm = pySaliencyMap.pySaliencyMap(img_width, img_height)


saliency_map = sm.SMGetSM(img)
binarized_map = sm.SMGetBinarizedSM(img)
salient_region = sm.SMGetSalientRegion(img)


# visualize
#    plt.subplot(2,2,1), plt.imshow(img, 'gray')
plt.subplot(2,2,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Input image')
#    cv2.imshow("input",  img)
plt.subplot(2,2,2), plt.imshow(saliency_map, 'gray')
plt.title('Saliency map')
#    cv2.imshow("output", map)
plt.subplot(2,2,3), plt.imshow(binarized_map)
plt.title('Binarilized saliency map')
#    cv2.imshow("Binarized", binarized_map)
plt.subplot(2,2,4), plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
plt.title('Salient region')
#    cv2.imshow("Segmented", segmented_map)

plt.show()
#    cv2.waitKey(0)
cv2.destroyAllWindows()