from sys import platform
import os
import cv2
import json
from pupil_apriltags import Detector
import open3d as o3d
from glob import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from Apriltag_Colmap import create_geometry_at_points, visualize_2d, colorbar
import collections
import k3d

def k3d_frustrum(pose, size=0.009, color=0x0000ff):
    # i.e. not using image sizes 
    pos = pose[:3, 3]
    
    forward = pose[:3, 2] * size * -1.4
    right = pose[:3, 0] * size * 1.25
    up = pose[:3, 1] * size
    
    #verts = [pos, pos + forward*size ]
    verts = [pos, pos + forward - right*0.5 + up*0.5, pos + forward + right * 0.5 + up * 0.5, pos ]
    verts += [pos, pos + forward - right*0.5 - up*0.5, pos + forward + right * 0.5 - up * 0.5, pos ]
    verts += [pos, pos + forward - right*0.5 + up*0.5, pos + forward - right * 0.5 - up * 0.5, pos]
    verts += [pos, pos + forward + right*0.5 + up*0.5, pos + forward + right * 0.5 - up * 0.5, pos]
    return k3d.line(verts, color=color, width=2.5, shader="simple")

def visualization_registration(reg, mesh):
    vertices = [ ]

    frustrums = []

    colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffcc00, 0xccff00, 0x00ccff]

    for i, frame in enumerate(reg.keys()):
        pos = reg[frame][:3, 3] 
        vertices += pos.tolist()
        frustrums.append( k3d_frustrum(reg[frame], size=0.1, color=colors[i % len(colors)]) )

    vertices = np.array(vertices)

    lines = k3d.line(vertices, color=0xff0000, width=2.5, shader="simple") # + line
    pts = k3d.points(vertices, point_size=0.003)

    plot3d = k3d.plot(antialias=True, camera_auto_fit=True)

    plot3d += lines
    plot3d += pts 

    for f in frustrums:
        plot3d += f

    plot3d += k3d.points( [vertices[:3]], point_size=0.0075, color=0x00ff00)

    plot3d += k3d.mesh(np.array(mesh.vertices), np.array(mesh.triangles).flatten(), color=0xffcc00)
    plot3d.display()

at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )


Register = collections.namedtuple(
    "RegisterInfo", ["CameraPose", "Intrinsic", "TagPose"])



if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":  
    # linux
        data_path  = "/home/biyang/Documents/3D_Gaze/dataset/3D_scanner_app/Apriltag1_dataset1"
        data_pi_path = "/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/office1/data_1"
        evaluation_json_path = r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/evaluation_apriltag_detection_Iphone.json"
    elif platform == "win32":
    # Windows...
        data_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\3D_Scanner_App\Apriltag1_dataset1"
        data_pi_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\office1\data_1"
        evaluation_json_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\evaluation_apriltag_detection_Iphone_a1d1.json"

    with open(evaluation_json_path, "r") as f:
        evaluation  = json.load(f)

    r = R.from_euler('xyz', [0, 180, 0], degrees=True)
    Additional_Rotation = r.as_matrix()
    additional_rotation = np.concatenate(
                         [np.concatenate([Additional_Rotation, np.array([[0], [0], [0]])], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    # Select one frame from iphone for each tag as register
    track_frame = {}
    for tag_id in evaluation.keys():
        distance_error = [alpha_error * 0.087 for alpha_error in evaluation[tag_id]["Error_Accuracy"]]
        distance_error = np.array(distance_error)
        
        index = np.argmin(distance_error)
        frame_id = evaluation[tag_id]["Frame_id"][index]

        track_frame[tag_id] = frame_id

    iphone_frames_folder= os.path.join(data_path, "frames")
    iphone_pose_folder = os.path.join(data_path, "pose")
    mesh_fullpath = os.path.join(data_path, "data3d/textured_output.obj")

    # Extract the camera pose and tag pose of each register
    register = {}

    for tag_id, frame_id in track_frame.items():

        camera_param_file = frame_id.replace(".jpg", ".json")
        img = cv2.imread(os.path.join(iphone_frames_folder, frame_id), cv2.IMREAD_GRAYSCALE)

        img_height, img_width = img.shape
        with open(os.path.join(iphone_pose_folder, camera_param_file), 'r') as f:
            camera_param = json.load(f)

        intrinsics = np.array(camera_param["intrinsics"]).reshape(3, 3)
        Cam2World = np.array(camera_param["cameraPoseARFrame"]).reshape(4, 4)

        fxfycxcy= [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
        tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = fxfycxcy, tag_size=0.087)

        for tag in tags:
            if tag.tag_id == int(tag_id):
                tag_pose = np.concatenate(
                        [np.concatenate([np.array(tag.pose_R), np.array(tag.pose_t)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
                register[tag.tag_id] = Register(CameraPose=Cam2World@additional_rotation, Intrinsic=intrinsics, TagPose=tag_pose)


    # Compute the Registration 
    # 1. Detect the tag in the frame
    # 2. Find the register with common tag
    # 3. Transform the new frame to the world coordinate
    
    PI_registration = {}
    PI_images_folder = os.path.join(data_pi_path, "images_undistorted")
    PI_intrinsics = np.array([[766.2927454396544, 0.0, 543.6272327745995],
                                [0.0, 766.3976103393867, 566.0580149497666],
                                [0.0, 0.0, 1.0]])
    PI_fxfycxcy= [PI_intrinsics[0, 0], PI_intrinsics[1, 1], PI_intrinsics[0, 2], PI_intrinsics[1, 2]]
    images = os.listdir(PI_images_folder)
    images.sort()

    for i, img_id in enumerate(images):
        if i % 100 != 0:
            continue

        img = cv2.imread(os.path.join(PI_images_folder, img_id), cv2.IMREAD_GRAYSCALE)

        PI_tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = PI_fxfycxcy, tag_size=0.087)
        if len(PI_tags) == 0:
            continue

        # Here I select simply one tag for the test
        PI_tag = PI_tags[0]

        Link_Register = register[PI_tag.tag_id]
        PI_tag_pose = np.concatenate(
                        [np.concatenate([np.array(PI_tag.pose_R), np.array(PI_tag.pose_t)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
        # PI_Cam2World
        PI_registration[int(img_id.split(".")[0])] =  Link_Register.CameraPose @ Link_Register.TagPose @ np.linalg.inv(PI_tag_pose)


    print(PI_registration)


    # Visualization of Camera Pose

    mesh_textured = o3d.io.read_triangle_mesh(mesh_fullpath, True)

    visualization_registration(PI_registration, mesh_textured)