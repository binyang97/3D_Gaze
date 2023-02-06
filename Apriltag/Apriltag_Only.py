from Apriltag.Apriltag_Colmap import *
from Apriltag.Apriltag_Test_CameraPose import *
from sys import platform


if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":  
    # linux
        data_path  = ""
    elif platform == "win32":
    # Windows...
        rawdata_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\2023_02_03_12_36_28"
        output_path= r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App"

