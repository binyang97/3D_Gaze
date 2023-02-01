import numpy as np
from sys import platform
from glob import glob
from load_ARSceneData import LoadARSceneData
import cv2
import open3d as o3d
import trimesh
import pyrender

if __name__ == "__main__":
    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
        traj_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.traj')[0]
        mesh_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.ply')[0]
        intrinsic_path = glob('/home/biyang/Documents/3D_Gaze/Colmap/' + ARKitSceneDataID + '/gt' + '/*.pincam')[0]

    elif platform == "win32":
        traj_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/lowres_wide.traj"
        mesh_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/40777060_3dod_mesh.ply"
        intrinsic_path = "D:/Documents/Semester_Project/Colmap_Test/40777060/GT/40777060_98.764.pincam"

    pose_gt, mesh_gt, intrinsics = LoadARSceneData(traj_path, mesh_path, intrinsic_path)

    print(pose_gt['163.255'])


    
    # Load the 3D mesh data
    mesh = trimesh.load(<path_to_your_3D_mesh>)

    # Define the camera intrinsic matrix
    K = np.array([[<camera_intrinsic_matrix>]])

    # Define the camera extrinsic matrix
    R = np.array([[<rotation_matrix>]])
    t = np.array([[<translation_vector>]])
    P = np.hstack((R, t))

    # Create a Pyrender scene
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])

    # Add the mesh to the scene
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)

    # Render the scene to a 2D image
    camera = pyrender.IntrinsicCamera(resolution=(<image_width>, <image_height>), intrinsic_matrix=K)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=0.5)
    camera_pose = np.hstack((R, t))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    rgb, _ = pyrender.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    # Optionally, apply additional image processing techniques
    <your_image_processing_code>
    Note: The example code is meant to be a simplified illustration of the steps involved in generating an RGB image from a 3D mesh. You may need to modify the code to suit your specific needs.




