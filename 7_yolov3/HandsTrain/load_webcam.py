import sys
import cv2 as cv
sys.path.append("../lib")
from utils.datasets import LoadWebcam
from infer_detector import Infer
infer=Infer(0)
classPath="classes.txt"
# weightsDir="../lib/weights/last_5epoch_2numgen.pt"
weightsDir="../lib/weights/last_5epoch_2numgen.pt"
infer.Model("yolov3", classPath, weightsDir, use_gpu=True, half_precision=True)
web=LoadWebcam()
iter(web)
while True:
    imgPath,img,img0,_=next(web)
    cv.imwrite("webcamFrames/frame.jpg", img0)
    infer.Predict("D:\\Repo\\Monk_Object_Detection\\7_yolov3\HandsTrain\\webcamFrames\\frame.jpg")
    out=cv.imread("output/frame.jpg")
    cv.imshow("frame", out)