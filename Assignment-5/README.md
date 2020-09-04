Human Pose Estimation

Architecture:-
Basically the model consists of Resnet-50 as the base and then 3 Deconvolution layers with kernal size=4 are applied to it.
JointMSE loss has been used as to compare the CNN obtained heatmaps and the target heatmaps.
The target heatmaps for joints are captured by applying the guassian centered on the joint ground truth.  



Original Image

![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-5/srkphotosddljpose.jpg)


with Human Pose Estimation

![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-5/srk.jpg)