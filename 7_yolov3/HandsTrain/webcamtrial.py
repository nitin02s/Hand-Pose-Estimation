import cv2 as cv
import os
import sys
import shutil
import time
sys.path.append("../lib")
from infer_detector import Infer
from utils.datasets import LoadWebcam

infer=Infer(0)
classPath="classes.txt"
weightsDir="../lib/weights/last_7e_2n.pt"
#weightsDir="../lib/weights/yolov3-tiny-8e-2n.pt"
infer.Model("yolov3", classPath, weightsDir, use_gpu=True)
# start=time.time()
# disp=2
# fps=0
cap=cv.VideoCapture(0)
prev=0
new=0
while True:
    ret, frame=cap.read()
    # frame=cv.flip(frame, 1)
    #frame=cv.resize(frame, (416, 416))
    print(frame.shape)
    cv.imwrite("webcamFrames/frame.jpg", frame)
    font=cv.FONT_HERSHEY_SIMPLEX
    new=time.time()
    fps=1/(new-prev)
    prev=new
    fps=int(fps)
    # if i%4==0:
    inputPath="D:\\Repo\\Monk_Object_Detection\\7_yolov3\HandsTrain\\webcamFrames\\frame.jpg"
    infer.Predict(inputPath)
    outputFrame=cv.imread("output/frame.jpg")
    cv.putText(outputFrame, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)
    # i+=1
    # os.system("cls")
    # print(f"FPS: {fps}")
    cv.imshow("frame", outputFrame)
    # cv.imshow("frame", frame)
    # shutil.rmtree("output")
    # shutil.rmtree("tmp")
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
# LoadWebcam()