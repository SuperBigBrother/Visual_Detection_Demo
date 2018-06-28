# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math


class Loophandler (object):
    def __init__(self, img):
        self.Image = img
        self.ImgArea = img.shape[0]*img.shape[1]
    

    #图像初始化，转换成CANNY边缘图
    def __convertImagetoCanny(self):
        self.ImageOriginal = self.Image
        if self.Image is None:
            print ('some problem with the image')
        else:
            print ('Image Loaded')
        self.Image = cv.cvtColor(self.Image, cv.COLOR_BGR2GRAY)
        self.Image = cv.GaussianBlur(self.Image, (5,5), 0)
        self.Image = cv.adaptiveThreshold(
            self.Image,
            255,                    # Value to assign
            cv.ADAPTIVE_THRESH_MEAN_C,# Mean threshold
            cv.THRESH_BINARY,
            11,                     # Block size of small area
            2,                      # Const to substract
        )
        self.Image = cv.Canny(
            self.Image,
            100,
            200,
        )
        self.WritingImage(self.Image, 'what')
        return self.Image
    
    
    #显示图像
    def WritingImage(self, image, imageName):
        if image is None:
            print ('Image is not valid.Please select some other image')
        else:
            cv.imshow(imageName, image)
            cv.waitKey(0)
            cv.destroyAllWindows()

    #找圆环，画圆环，只用了最简单的算法
    def LoopFinding(self):
        thresholdImage = self.__convertImagetoCanny()
        circles = cv.HoughCircles(
            thresholdImage,
            cv.HOUGH_GRADIENT, 1, 50,
            param1 = 50, param2 = 50,
            minRadius = 0, maxRadius = 0,
        )
        for cir in circles[0]:
            cv.circle(self.ImageOriginal, (cir[0], cir[1]), cir[2], (0,255,0), 2)
            cv.circle(self.ImageOriginal, (cir[0], cir[1]), 2, (0,0,255), 3)
        self.WritingImage(self.ImageOriginal, 'circles')
        print(circles)