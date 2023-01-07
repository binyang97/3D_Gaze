import json 
import numpy as np
from scipy.optimize import minimize
import open3d as o3d
import copy
from math import sqrt
from sys import platform
from glob import glob

def draw_registration_result(source, target, transformation, colored = True):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if colored:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def rigid_transform_3D(A, B, scale):

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)


    t_diff = centroid_B - centroid_A

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T 

    return c, R, t


class TransfromationExtractor:
    def __init__(self, json_path_reconstruction, json_path_gt, scaling):
        with open(json_path_gt) as json_file:
            self.dict_gt = json.load(json_file)

        with open(json_path_reconstruction) as json_file:
            self.dict_rc = json.load(json_file)

        self.keypoints_gt = self.GetPointList(self.dict_gt)
        self.keypoints_rc = self.GetPointList(self.dict_rc)

        self.scaling = scaling

    @staticmethod
    def GetPointList(keypoints_dict):
        points = []
        for i, cls in enumerate(keypoints_dict["keypoints"]):
            points.append(cls["position"])

        points = np.matrix(points)
        
        return points
            
    def check(self):

        try:
            assert len(self.keypoints_gt) == len(self.keypoints_rc) 
        except:
            raise AssertionError("The size of points are not matched")

    def extract(self):
        self.s_opt, self.R_opt, self.t_opt = rigid_transform_3D(self.keypoints_gt, self.keypoints_rc, self.scaling)

    def compute_error(self):
        n = self.keypoints_rc.shape[0]
        if self.scaling:
            transformed_rc = self.keypoints_rc * self.s_opt
        else:
            transformed_rc = self.keypoints_rc
        transformed_rc = (self.R_opt * transformed_rc.T) + np.tile(self.t_opt, (1, n))
        self.transformed_rc = transformed_rc.T

        
        err = self.keypoints_gt - self.transformed_rc
        err = np.multiply(err, err)
        err = np.sum(err)
        self.rmse = sqrt(err / n)


        print("The RMSE error is: ", self.rmse)  
    

    
if __name__ == "__main__":

    path_json_gt = "/home/biyang/Documents/3D_Gaze/Colmap/40777060/pointcloud/gt_100000.json"
    path_json_rc = "/home/biyang/Documents/3D_Gaze/Colmap/40777060/pointcloud/reconstruction_100000.json"

    extractor = TransfromationExtractor(path_json_rc, path_json_gt, scaling = True)
    extractor.check()

    extractor.extract()

    extractor.compute_error()

    VIS_KEYPOINTS = True
    TEST = False

    est_extrinsic = np.concatenate(
                    [np.concatenate([extractor.R_opt, extractor.t_opt.reshape(3, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    if VIS_KEYPOINTS:

        pcd_gt= o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(extractor.keypoints_gt)

        pcd_rc= o3d.geometry.PointCloud()
        pcd_rc.points = o3d.utility.Vector3dVector(extractor.keypoints_rc)

        pcd_rc.scale(extractor.s_opt ,center=np.zeros(3))

        draw_registration_result(pcd_rc, pcd_gt, est_extrinsic)

    ARKitSceneDataID = "40777060"
    if platform == "linux" or platform == "linux2":  
    # linux
        path_reconstruction = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/output/0/meshed-poisson.ply")
        path_gt = glob("/home/biyang/Documents/3D_Gaze/Colmap/" + ARKitSceneDataID + "/gt/*.ply")

    elif platform == "win32":
    # Windows...
        path_gt = glob('D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/GT/*.ply')
        path_reconstruction = glob('D:/Documents/Semester_Project/Colmap_Test/' + ARKitSceneDataID + '/Output/meshed-poisson.ply')


    mesh_reconstruction = o3d.io.read_triangle_mesh(path_reconstruction[-1])
    mesh_gt = o3d.io.read_triangle_mesh(path_gt[-1])

    if TEST:
        mesh_reconstruction.scale(extractor.s_opt, center = np.zeros(3))
        draw_registration_result(mesh_reconstruction, mesh_gt, est_extrinsic, colored=False)




