--> Gaze Estimation via Egocentric rgb images and Saliency Maps:
https://openaccess.thecvf.com/content_ECCV_2018/papers/Huang_Predicting_Gaze_in_ECCV_2018_paper.pdf
https://github.com/hyf015/egocentric-gaze-prediction


--> Gaze Heat Map Estimation: https://arxiv.org/pdf/2006.00626.pdf
The paper also gives a short benchmark of different FPV datasets 
-- did not find the source code


GTEA Gaze+: no access to the dataset rightnow

-->Deep Future Gaze: Gaze Anticipation on Egocentric Videos
Using Adversarial Networks:  Gaze Estimation via Adversarial Networks (Gaze Anticipation)
https://github.com/Mengmi/deepfuturegaze_gan 
-- Not a good idea to use, GAN is too complicated for further feature feeding


In the Eye of Transformer: Global-Local Correlation for Egocentric Gaze Estimation
--> https://bolinlai.github.io/GLC-EgoGazeEst/
This one has the most highest score in the paper (but the source code is not found yet)
3.9% on EGTEA Gaze+ and 5.6% on Ego4D

Here is the question: how to show the improvement? The baseline production is based on existing approach but our custom dataset


Gaze Estimation + Action Recognition --> Multi-task Network
Mutual Context Network for Jointly Estimating
Egocentric Gaze and Action
https://arxiv.org/pdf/1901.01874.pdf