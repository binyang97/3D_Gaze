# Camera Localization for 3D gaze estimation
The repository is the conclusion of my work for the semester project

## Environmental Setup
```
conda install -c conda-forge colmap
pip install -r requirements 
```
You can also clone the original repository of colmap to use the GUI functionalities:
https://github.com/colmap/colmap


## Reference
AprilTag: https://github.com/AprilRobotics/apriltag \
ARKitScenes: https://github.com/apple/ARKitScenes \
Pupil Invisible: https://pupil-labs.com/products/invisible/ 


## File Structures
```
📦3D_Gaze
 ┣ 📂3D_Scanner  # File cleaner for data recorded from 3D Scanner App
 ┃ ┗ 📜file_clearup.py
 ┣ 📂Apriltag    # Evaluation for Apriltag detection on 2D images and PnP-Pose Estimator
 ┃ ┣ 📜Evalutation_Arpiltag_Detection.py
 ┃ ┣ 📜Evalutation_Arpiltag_Detection_2.py
 ┃ ┗ 📜__init__.py
 ┣ 📂docs       # Documentations
 ┃ ┣ 📂presentations
 ┃ ┃ ┣ 📜Präsentation1.pptx
 ┃ ┃ ┣ 📜Präsentation_report.pptx
 ┃ ┃ ┣ 📜SP_Status.potx
 ┃ ┃ ┗ 📜SP_Status.pptx
 ┃ ┣ 📂Proposal
 ┃ ┃ ┣ 📂Figures
 ┃ ┃ ┃ ┣ 📜sp_overview.pptx
 ┃ ┃ ┃ ┗ 📜sp_paper_reading.pptx
 ┃ ┃ ┣ 📂ref_paper
 ┃ ┃ ┃ ┗ 📂data_collection
 ┃ ┃ ┃ ┃ ┗ 📜1601.02644.pdf
 ┃ ┃ ┣ 📜SP_Proposal.pptx
 ┃ ┃ ┣ 📜SP_Proposal_V01.pdf
 ┃ ┃ ┗ 📜SP_Proposal_V02.pdf
 ┃ ┗ 📜SP_FinalReport_TypoChecked.pdf
 ┣ 📂functions        # Util-functions
 ┃ ┣ 📜depth_extraction.py
 ┃ ┣ 📜pySaliencyMap.py
 ┃ ┣ 📜pySaliencyMapDefs.py
 ┃ ┣ 📜test_rgb_registration.py
 ┃ ┗ 📜trimesh_new.py
 ┣ 📂PoseEstimation    # Codes for pose estimation on different dataset
 ┃ ┣ 📂Archiv
 ┃ ┣ 📜Apriltag_Colmap.py
 ┃ ┣ 📜Apriltag_Registration.py
 ┃ ┣ 📜Apriltag_Test_CameraPose.py
 ┃ ┣ 📜Apriltag_Test_Filter_Keypoints.py
 ┃ ┣ 📜Apriltag_Visualization.py
 ┃ ┣ 📜camera_pose_visualizer.py
 ┃ ┣ 📜Colmap_Reader.py
 ┃ ┣ 📜colmap_read_write_model.py
 ┃ ┣ 📜create_rgb_from_pcd.py
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜evaluate_apt0.py
 ┃ ┣ 📜evaluate_ARSceneData.py
 ┃ ┣ 📜GT_Extration.py
 ┃ ┣ 📜load_ARSceneData.py
 ┃ ┣ 📜Mesh_Matcher.py
 ┃ ┣ 📜prepare_data_ARScene.py
 ┃ ┣ 📜Rendererd_Image_apritag_detection.py
 ┃ ┣ 📜Test_Ray_Casting.py
 ┃ ┗ 📜__init__.py
 ┣ 📂preprocessing_pi_data   # Preprocessing codes for data recoded by Pupil Invisible
 ┃ ┣ 📂ref_paper
 ┃ ┣ 📜data_preprocessing.py
 ┃ ┣ 📜image_noise_filtering.py
 ┃ ┣ 📜image_preprocessing.py
 ┃ ┗ 📜Readme.md
 ┣ 📂test_dataset
 ┃ ┣ 📂depth
 ┃ ┣ 📂pose
 ┃ ┗ 📂rgb
 ┣ 📜.gitignore
 ┣ 📜Apriltag_Registration_Pipeline_Experiment - Kopie.ipynb   # Apriltag-framework including Visualization
 ┣ 📜COLMAP_Registration_Pipeline_Experiment.ipynb             # Colmap-framework including Visualization
 ┣ 📜DepthMap_Extraction.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┣ 📜SailencyMap_Extraction.py
 ┗ 📜__init__.py
```
