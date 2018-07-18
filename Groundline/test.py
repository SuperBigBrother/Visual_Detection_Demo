# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np 

cap = cv.VideoCapture(0)

while True:
    if cap.isOpened():
        ret, frame = cap.read()

        cv.imshow('frame',frame)

        cv.waitKey(20)

cap.release()
cv.destroyAllWindows()