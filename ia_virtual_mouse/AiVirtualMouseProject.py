import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##################################
wCam, hCam = 640, 480
##################################

cap = cv2.VideoCapture("http://192.168.0.5:4747/video")
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    success, img = cap.read()
    
    if success:
        cv2.imshow('image', img)
        cv2.waitKey(1)