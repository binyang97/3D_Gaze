from pupil_apriltags import Detector, Detection
from sys import platform
from glob import glob
import cv2


if __name__ == '__main__':

    if platform == "linux" or platform == "linux2":  
    # linux
        dataset_path = r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/room1/images"

    elif platform == "win32":
    # Windows...
        dataset_path = None


    image_list = glob(dataset_path + "/*.jpg")
    image_list.sort()
    index = 394
    
    img = cv2.imread(image_list[index], cv2.IMREAD_GRAYSCALE)

    at_detector = Detector(
    families="tagStandard41h12",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
    )

    tags = at_detector.detect(img)
    print(tags[0].corners)


    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

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