import os
import numpy as np
import json
import open3d as o3d
from Apriltag.Apriltag_Colmap import *
from sklearn.cluster import DBSCAN
from numpy import unique



if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\image_100_undistorted_prerecorded\Stereo_Fusion.min_num_pixels=10\dense\fused.ply")
    pcd.paint_uniform_color(np.array([210, 210, 210])/255)

    with open(r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\apriltags.json", "r") as outfile:
       dic = json.load(outfile)

    with open(r"D:\Documents\Semester_Project\3D_Gaze\dataset\PupilInvisible\room1\apriltags_repro_error.json", "r") as outfile:
       dic2 = json.load(outfile)

    registered_points = {int(k): np.array(v) for k, v in dic.items()}
    registered_points_repro_error = {int(k): np.array(v) for k, v in dic2.items()}


    #print(registered_points_repro_error)
    


    outliers = None

    # Parameter
    #min_repro_error = 1.1

    for key in registered_points.keys():
        point_cloud = registered_points[key]
        repro_error = registered_points_repro_error[key]

        min_repro_error = np.mean(repro_error) - 0*np.std(repro_error)
        print(min_repro_error)

        valid_repro_error = np.zeros(len(repro_error), dtype=bool)
        valid_repro_error[repro_error < min_repro_error] = True

        outliers_too_large_repro_error = point_cloud[~valid_repro_error]
        point_cloud = point_cloud[valid_repro_error]
        
        core_samples_mask = np.zeros(len(point_cloud), dtype=bool)
        # Define the DBSCAN function
        # eps is the most important parameters
        # minimum distance that will be defined as a neighbor
        dbscan = DBSCAN(eps=0.5, min_samples=5)

        # Fit the model to the point cloud data
        dbscan.fit(point_cloud)
        
        labels = dbscan.labels_
        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        counts = counts[unique_labels >= 0]
        unique_labels = unique_labels[unique_labels >= 0]

        #print(unique_labels, counts, key)
        # Case when there is multiple valid clustered classes --> simple idea: directly take the one with the most count
        if len(unique_labels) > 1:
            main_label = unique_labels[np.argmax(counts)]
            core_samples_mask[labels == main_label] = True
        else:
            core_samples_mask[dbscan.core_sample_indices_] = True

        registered_points[key] = point_cloud[core_samples_mask]

        if outliers is None:
            outliers = point_cloud[~core_samples_mask]
        else:
            outliers = np.concatenate((outliers, point_cloud[~core_samples_mask]), axis=0)

        if outliers is None:
            outliers = outliers_too_large_repro_error
        else:
            outliers = np.concatenate((outliers, outliers_too_large_repro_error), axis=0)
    
    pcd_outliers = create_pcd(outliers, color = [0, 0 ,0])
    vis = [pcd_outliers]
    for i, tag_id in enumerate(registered_points.keys()):
    
        pcd_tag = create_pcd(registered_points[tag_id], colorbar[i])
        vis.append(pcd_tag)

    #o3d.visualization.draw_geometries(vis)

    center_points = []
    for points in registered_points.values():
        center_point = np.mean(points, axis=0)
        center_points.append(center_point)

    pcd_center = create_geometry_at_points(np.array(center_points), color = [0, 0,0], radius = 0.5)
    o3d.visualization.draw_geometries([pcd_center])

    # print(np.array(center_points))

    print(registered_points.keys())

    print(np.mean(registered_points[0], axis=0))
    print(np.mean(registered_points[2], axis=0))
    print(np.mean(registered_points[5], axis=0))
    print(np.mean(registered_points[3], axis=0))
    print(np.mean(registered_points[1], axis=0))
    print(np.mean(registered_points[4], axis=0))


    