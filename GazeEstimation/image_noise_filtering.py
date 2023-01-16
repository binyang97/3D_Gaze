import cv2
import os
from sys import platform
import matplotlib.pyplot as plt
from glob import glob

if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":  
    # linux
        pass

    elif platform == "win32":
    # Windows...
        pi_imagepath = glob(r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1_v2\images" + "/*")
        iphone_imagepath = glob(r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\TestPhone\images" + "/*")

    pi_imagepath.sort()
    iphone_imagepath.sort()

    id = 20

    image = cv2.imread(pi_imagepath[id],1) 

    #image_bw = cv2.imread(pi_imagepath[id],0) 


    #noiseless_image_bw = cv2.fastNlMeansDenoising(image_bw, None, 20, 7, 21) 

    noiseless_image_colored = cv2.fastNlMeansDenoisingColored(image,None,20,20,7,21) 

    gaussian_blur = cv2.GaussianBlur(image,(9,9),0)
    median_blur = cv2.medianBlur(image,9)
    bilateral_blur = cv2.bilateralFilter(image,15,75,75)





    titles = ['Original Image(colored)','Gaussian Blurring', 'Mediann Blurring', 'Bilateral Blurring']
               # 'Original Image (grayscale)','Image after removing the noise (grayscale)']
    images = [image, gaussian_blur, median_blur, bilateral_blur]
    plt.figure(figsize=(13,5))
    for i in range(len(images)):
        plt.subplot(2,2,i+1)
        plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
