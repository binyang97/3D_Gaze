import numpy as np
from sys import platform
from glob import glob
from load_ARSceneData import LoadARSceneData
import cv2
import open3d as o3d
import trimesh
import pyrender
import re
import os
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
        traj_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.traj')[0]
        mesh_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.ply')[0]
        intrinsic_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.pincam')[0]
        image_path = "/home/biyang/Documents/3D_Gaze/dataset/ARKitScenes/3dod/Training/40777060/40777060_frames/lowres_wide"


        mesh_path = "/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_phone/dense/0/meshed-poisson.ply"

    elif platform == "win32":
        traj_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/lowres_wide.traj"
        mesh_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/40777060_3dod_mesh.ply"
        intrinsic_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/40777060_98.764.pincam"

        image_path = r""

        
    # index = 30
    # poses_gt, _, K = LoadARSceneData(traj_path, mesh_path, intrinsic_path)

    # marker1 = '_'
    # marker2 = '.png'
    # regexPattern = marker1 + '(.+?)' + marker2


    # images = os.listdir(image_path)
    # image_filename = images[index]
    # frame_id = re.search(regexPattern, image_filename).group(1)
    # pose = np.array(poses_gt[frame_id])


    # Translation = pose[:3, 3].reshape(3, 1)
    # Rotation = pose[:3, :3]
    # T_inv = np.matmul(-Rotation.T, Translation)
    # R_inv = Rotation.T
    # RT_inv = np.concatenate(((R_inv, T_inv)), axis=-1)
    # world2cam = np.vstack((RT_inv, np.array([0, 0, 0, 1])))

    # img = cv2.imread(os.path.join(image_path, image_filename), cv2.COLOR_BGR2RGB)

    # Create a renderer with a set image width and height

    SAVE_IMG = False
    output_path = "/home/biyang/Documents/3D_Gaze/Colmap/PI_room1/Test_phone/Test_Rendering"

    camera_params = [766.2927454396544, 766.3976103393867, 543.6272327745995, 566.0580149497666]

    image_name = "Test4.jpg"

    K = np.array([[camera_params[0], 0, 320],
                    [0, camera_params[1], 240],
                    [0, 0, 1]])

    render = rendering.OffscreenRenderer(640, 480)

    # setup camera intrinsic values
    pinhole = o3d.camera.PinholeCameraIntrinsic(640, 480, camera_params[0], camera_params[1], 320, 240)
        
    # Pick a background colour of the rendered image, I set it as black (default is light gray)
    render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

    # now create your mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # The default camera coordinate of open3d and ARKit Scene are different
    r = R.from_euler('xyz', [-60, 180, 0], degrees=True)
    Additional_Rotation = r.as_matrix()
    est_extrinsic = np.concatenate(
                    [np.concatenate([Additional_Rotation, np.zeros((3, 1))], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    #mesh.transform(world2cam)
    mesh.transform(est_extrinsic)
    #mesh.paint_uniform_color([1.0, 0.0, 0.0]) # set Red color for mesh 
    # define further mesh properties, shape, vertices etc  (omitted here)  

    # Define a simple unlit Material.
    # (The base color does not replace the mesh's own colors.)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # add mesh to the scene
    render.scene.add_geometry("MyMeshModel", mesh, mtl)
    #render.scene.add_geometry(mesh)

    # render the scene with respect to the camera
    render.scene.camera.set_projection(K, 0.1, 1.0, 640, 480)
    img_o3d = render.render_to_image()

    o3d.visualization.draw_geometries([img_o3d])

    
    
    img_rendered = np.asarray(img_o3d)
    img_rgb = img_rendered[...,::-1].copy()

    if SAVE_IMG:
        cv2.imwrite(os.path.join(output_path, image_name), img_rgb)
    # Visualization
    # vis = np.concatenate((img, img_rendered), axis = 1)

    # cv2.imshow('HORIZONTAL', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optionally, apply additional image processing techniques


