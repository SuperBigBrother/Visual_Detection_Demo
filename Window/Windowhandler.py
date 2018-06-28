# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math



class Windowhandler (object):
    def __init__(self, img):
        self.Image = img
        self.ImgArea = img.shape[0]*img.shape[1]
    
    #转换成CANNY图
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

        return self.Image

    #显示图片
    def WritingImage(self, image, imageName):
        if image is None:
            print ('Image is not valid.Please select some other image')
        else:
            cv.imshow(imageName, image)
            cv.waitKey(0)
            cv.destroyAllWindows()
    
    #过滤轮廓，首先满足轮廓大小大于某个值，并保证为封闭多边形
    def ContoursFilter(self, contours):
        areasize_max = 0.75 * self.ImgArea
        areasize_min = 0.01 * self.ImgArea
        new_contours = []
        for cont in contours:
            if cv.isContourConvex(cont) is False and cv.contourArea(cont)>areasize_min :
                new_contours.append(cont)
        return new_contours

    #获得轮廓
    def GetImageContour(self):
        thresholdImage = self.__convertImagetoCanny()
        thresholdImage, contours, hierarchy =cv.findContours(
            thresholdImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE,
        )
        contours = self.ContoursFilter(contours)
        self.Contours = contours
        Drawcontours = cv.drawContours(self.ImageOriginal, self.Contours, -1, (0,0,255), 3)
        return contours,Drawcontours

    #计算两条直线的夹角，pt0-pt2为矩形的任意三个顶点
    def angle(self, pt1, pt2,pt0):
        dx1 = pt1[0] - pt0[0]
        dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]
        dy2 = pt2[1] - pt0[1]
        angle_line = (dx1*dx2 + dy1*dy2) / (math.sqrt(dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10)
        return math.acos(angle_line)*180/3.141592653

    #计算矩形中心，pt0-pt2为矩形任意三个定点
    def center(self, pt0, pt1, pt2):
        pt = np.array([pt0, pt1, pt2])
        pt_sum = np.zeros(3)
        pt_sum[0] = pt0[0] + pt0[1]
        pt_sum[1] = pt1[0] + pt1[1]
        pt_sum[2] = pt2[0] + pt2[1]
        idx = np.argmin(pt_sum)
        dx = pt[idx-1][0] - pt[idx][0]
        dy = pt[idx-2][1] - pt[idx][1]
        center = np.array([int(pt[idx][0]+dx/2),int(pt[idx][1]+dy/2)])
        return center

    #检测直角
    def Checkangles(self, contours, delta = 0.7, epsilon = 0.05, minangle = 80, maxangle = 100):
        rectangle = []
        center = []
        for con in contours:
            approx = cv.approxPolyDP(con, epsilon * cv.arcLength(con, True), True)
            # if shape is a rectangle approx.shape = (4,1,2)
            if approx.shape[0] == 4:
                angle = np.zeros(3)
                for i in range(2,5,1):
                    t = math.fabs(self.angle(approx[i%4][0], approx[i-2][0], approx[i-1][0]))
                    angle[i-2] = t   
                if max(angle) < maxangle and min(angle) > minangle:
                    center.append(self.center(approx[i%4][0], approx[i-2][0], approx[i-1][0]))
                    rectangle.append(con)
        return rectangle, center
    
    #找窗户矩形
    def WindowFinding(self):
        contours, Drawcontours = self.GetImageContour()
        self.WritingImage(Drawcontours, 'contours')
        if len(contours) == 0:
            print ('no rectangle in image')
        else:
            rectangle, center = self.Checkangles(contours)
            if len(rectangle) > 0 :
                self.findrectangle = True
                print ('the center of rectangle is  : {}'.format(center))
                Drawrectangles = cv.drawContours(self.ImageOriginal, rectangle, -1, (0,0,255), 3)
                self.WritingImage(Drawrectangles, 'rectangle')
            else:
                self.findrectangle = False
    #判断是否找到
    def WindowInImage(self):
        return self.findrectangle
        