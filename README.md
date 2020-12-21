# Natural-user-interface
Using your hands as a medium to interact with a computer or any machine




## Project Details

# YOLO

7_yolov3 uses a yolo model for detection of hand. The models has been on a custom data set which were annotated manually for our requirement. The pipeline for this model is present at  - https://github.com/ultralytics/yolov3

Webcamtrial.py can be used for real time inference using your laptop webcam where the weights acquired from training the model can be loaded at the specific lines of code.

# 21 POINTS

The second folder i.e., key points folder consists of the code pertaining to training and inferring 21 key points present in the hand which is used to visualize the skeleton of the hand. The data set used was freinhand dataset which can be found here - Computer Vision Group, Freiburg (uni-freiburg.de). The dataset is a 3D dataset. I have converted the 3D dataset to 2D for initial understanding purposes. Training_xy.csv consists of XY positions for all the 21 points for each image. The model used was Resnet whose architecture is present in model.py. notebook2-final.ipynb is a notebook for training the model and inferring the result from the model. The acquired results are present in the predicted picture folder.



