import numpy as np
from pupil_apriltags import Detector
from sys import platform
from glob import glob
import cv2
import collections
from typing import List, Tuple, Dict
from Apriltag import colorbar, create_pcd
import open3d as o3d
from GT_Extration import rigid_transform_3D, draw_registration_result
from scipy.optimize import minimize

def transform_3d(points1, points2):
    def objective(x):
        # Extract the rotation matrix, translation vector, and scale factor from the optimization variables
        R = np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]])
        t = np.array([x[9], x[10], x[11]])
        s = x[12]

        # Transform the points using the rotation matrix, translation vector, and scale factor
        transformed_points = s * np.dot(points1, R.T) + t

        # Calculate the sum of squared differences between the transformed points and the target points
        return np.sum((transformed_points - points2)**2)

    # Set up the optimization constraints
    cons = ({'type': 'eq', 'fun': lambda x: np.array([x[0]**2 + x[3]**2 + x[6]**2 - 1,
                                                    x[1]**2 + x[4]**2 + x[7]**2 - 1,
                                                    x[2]**2 + x[5]**2 + x[8]**2 - 1])},
            {'type': 'ineq', 'fun': lambda x: x[12]})

    # Minimize the objective function subject to the constraints
    result = minimize(objective, np.zeros(13), method='SLSQP', constraints=cons)

    # Extract the optimized rotation matrix, translation vector, and scale factor
    R_opt = np.array([[result.x[0], result.x[1], result.x[2]],
                    [result.x[3], result.x[4], result.x[5]],
                    [result.x[6], result.x[7], result.x[8]]])
    t_opt = np.array([result.x[9], result.x[10], result.x[11]])
    s_opt = result.x[12]

    return s_opt, R_opt, t_opt


TagPose = collections.namedtuple(
    "Pose", ["tag_id", "R", "t", "error"])

RelativeTransform = collections.namedtuple(
    "Pose", ["description", "R", "t"])

def compute_transformation_matrix(transformation_order: List[int], all_tag_poses: List[List[Tuple]], common_tag_ids: List[int]) -> np.ndarray:

    # Initialization
    Rotation = np.eye(3)
    Translation = np.zeros((3, 1))
    for i in range(len(transformation_order)-1):
        source_frame_id = transformation_order[i]
        target_frame_id = transformation_order[i+1]

        common_tag_id = common_tag_ids[i]

        source_poses = all_tag_poses[source_frame_id]
        target_poses = all_tag_poses[target_frame_id]

        source_pose = find_tag_pose(source_poses, common_tag_id)
        target_pose = find_tag_pose(target_poses, common_tag_id)

        R_transform = target_pose.R.T @ source_pose.R
        t_transform = target_pose.R.T @ source_pose.t - target_pose.R.T @ target_pose.t

        Rotation = R_transform @ Rotation
        Translation = R_transform @ Translation + t_transform

    return Rotation, Translation



def find_tag_pose(tag_poses, target_tag_id):
    for pose in tag_poses:
        if pose.tag_id == target_tag_id:
            return pose
        
    raise ValueError("Could not find tag pose for target tag id {}".format(target_tag_id))
    
def dfs(graph, start, end, path):
    path.append(start)
    if start == end:
        return path
    for node in graph[start]:
        if node not in path:
            new_path = dfs(graph, node, end, path[:])
            if new_path:
                return new_path
    return None

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False



if __name__ == "__main__":
    
    if platform == "linux" or platform == "linux2":  
    # linux
        images_path = r"/home/biyang/Documents/3D_Gaze/dataset/PupilInvisible/room1/images_gt_apriltags_undistorted"

    elif platform == "win32":
    # Windows...
        images_path = r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\images_gt_apriltags_undistorted"
        
    camera_params = [766.2927454396544, 766.3976103393867, 543.6272327745995, 566.0580149497666]
    at_detector = Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )


    image_list = glob(images_path+ "/*.jpg")



    tags_in_images = []
    tags_id_in_images = []

    graph = {}

    for i, image_fullpath in enumerate(image_list):
    
        img = cv2.imread(image_fullpath, cv2.IMREAD_GRAYSCALE)
        tags = at_detector.detect(img, estimate_tag_pose=True, camera_params = camera_params, tag_size=0.01)
        tags_in_image = []
        tags_id_in_image = []
        for tag in tags:
            tags_in_image.append(TagPose(tag_id=tag.tag_id, 
                                        R = tag.pose_R, 
                                        t = tag.pose_t, error = tag.pose_err))
            
            tags_id_in_image.append(tag.tag_id)
        tags_in_images.append(tags_in_image)
        tags_id_in_images.append(tags_id_in_image)

    # tag_ids = set([item for sublist in tags_id_in_images for item in sublist])

    # print(tags_id_in_images)

    graph = {}
    end_node = 0
    num_tags = 0
    for i, img in enumerate(tags_id_in_images):
        if len(img) > num_tags:
            end_node = i
            num_tags = len(img)
        if i in graph.keys():
            pass
        else:
            graph[i] = []
        for j,  other_image in enumerate(tags_id_in_images):
            if i == j:
                continue
            else:
                if common_member(img, other_image) and j not in graph[i]:
                    graph[i].append(j)

    # Get a path for  each frame if we want to transform the coordinate to the same reference coordinate (end_node)
    paths = []
    for start_node in range(len(image_list)):
        paths.append(dfs(graph = graph, start=start_node, end=end_node, path = []))

    # For each path, find the common tag_id to compute the relative transformation matrix
    common_tag_ids = []
    for path in paths:
        if len(path) == 1:
            common_tag_id = None
        else:
            common_tag_id = []
            for i in range(1, len(path)):
                common_tag_id.append(list(set(tags_id_in_images[path[i]]).intersection(tags_id_in_images[path[i-1]]))[-1])
            
        common_tag_ids.append(common_tag_id)

    print(tags_id_in_images)
    print(common_tag_ids)
    print(paths)
    

    # Compute the relative transformation matrix with given path and common tag_id pairs
    relative_transformations = []
    
    for path, tag_id in zip(paths, common_tag_ids):
        R_rel, T_rel = compute_transformation_matrix(path, tags_in_images, tag_id)

        relative_transformations.append(RelativeTransform(description=path, R=R_rel, t=T_rel))

    # Utilize the coordinate with transformation matrix
    utilized_tags_xyz = {}
    for tag in tags_in_images[end_node]:
        utilized_tags_xyz[tag.tag_id] = -tag.R.T @ tag.t

    for transformation, tags_in_image in zip(relative_transformations, tags_in_images):
        for tag_pose in tags_in_image:
            if tag_pose.tag_id in utilized_tags_xyz.keys():
                continue
            else:
                utilized_tags_xyz[tag_pose.tag_id] = transformation.R @ (-tag_pose.R.T @ tag_pose.t) + transformation.t


    points = np.array([v for v in utilized_tags_xyz.values()])

    print(utilized_tags_xyz.keys())

    gt_points = points.reshape(points.shape[0], points.shape[1])
    gt_points = np.asmatrix(gt_points)
    

    rc_points = np.matrix([[-28.27302669 , -7.87766168 , 22.62987562],
                            [ -8.83791871 ,  1.77397253,  11.4245153 ],
                            [ -8.49405983, -11.76506922,  19.18423766],
                            [-15.63503554 ,  1.76817988,  -6.65658561],
                            [  7.18639042 ,  3.90690051,  15.25967566],
                            [  6.40895162,  15.52132884 , -1.40985782]])

    s_opt, R_opt, t_opt = rigid_transform_3D(gt_points, rc_points, scale=True)
    #s_opt, R_opt, t_opt = transform_3d(np.asarray(rc_points), np.asarray(gt_points))

    print(t_opt)

    est_extrinsic = np.concatenate(
                    [np.concatenate([R_opt, t_opt.reshape(3, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

    VIS_KEYPOINTS = True

    print(s_opt)

    if VIS_KEYPOINTS:

        pcd_gt= o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_points)

        pcd_rc= o3d.geometry.PointCloud()
        pcd_rc.points = o3d.utility.Vector3dVector(rc_points)

        pcd_rc.scale(s_opt ,center=np.zeros(3))

        draw_registration_result(pcd_rc, pcd_gt, est_extrinsic)

    #print(points)
    #pcd = create_pcd(points=points, color=[0, 0 ,0])

    #o3d.visualization.draw_geometries([pcd])

    

    




    
    
        

     

    

    