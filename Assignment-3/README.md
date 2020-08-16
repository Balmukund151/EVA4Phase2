
**Table of Contents**

Link to the static website for Resnet34, Customized MobileNetV2, Face Align and Face Swap

http://model-mobilenetv2.s3-website.ap-south-1.amazonaws.com/

# Resnet34 with AWS Deployment Code

AWS-resnet34 folder contains the jit traced saved model for resnet34.
The AWS code is same as of Assignment-1

# Customized MobilenetV2 on Drone & Bird with AWS Deployment Code

The MobileNetV2 model is trained on 'Flying Birds' , 'Large QuadCopters' , 'Small QuadCopters' & 'Winged Drones'.

Labels are as below:-
'Flying Birds'=0 , 'Large QuadCopters'=1 , 'Small QuadCopters'=2 , 'Winged Drones'=3


"AWS-custom-mobilenetv2" contains the code for deploying the Customized MobileNetV2 model on AWS.

# Face Alignment with AWS Deployment Code

The idea here is that a face pic in some angle is uploaded and using Dlib face is aligned so that it is focusing front.

i have tested it by uploading a face of "apjabdul-kalam"  and aligning it to focus in front.

Original Face

![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-3/AWS-face-Align/apjabdul-kalam.jpg)

Aligned Face

![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-3/AWS-face-Align/apjabdul-kalam-Aligned.jpg)

"AWS-face-Align" folder contains the code for Deploying the face-Alignment code on AWS.

# Face Swap with AWS Deployment Code

The idea here is to swap a face with another face.
It takes 2 faces as input and overlays the 1st face on top of another face.

i have tried to swap Mr.Nitesh Kumar face on top of Mr. Modi

Mr. Nitesh Kumar & Mr. Modi (Original Face)

![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-3/AWS-face-swap/nitish-kumar-and-Modi.jpg)


Mr. Nitesh Kumar face layed on top of Mr.Modi

![alt text](https://github.com/Balmukund151/EVA4Phase2/blob/master/Assignment-3/AWS-face-swap/nitish-kumar-on-Modi.jpg)

"AWS-face-swap" folder contains the code to deploy it on AWS.

