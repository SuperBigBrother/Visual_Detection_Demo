# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np 

#Capture Video Initialize
cap = cv.VideoCapture(0)
GRAYSCALE_THRESHOLD = 100    #黑线[0,64],白线[128,255]
ROI_h = 50
ROIS = [[0,430,640,ROI_h,5],
        [0,315,640,ROI_h,4],
        [0,215,640,ROI_h,3],
        [0,115,640,ROI_h,2],
        [0,0,640,ROI_h,1]]    #[x,y,w,h,index]




#找到黑块，blob_h 为高度阈值，blob_w 为宽度阈值
#blob 为一个列表类型[块的起始点， 块的长度， ROI序号]第三个参数暂时还没有用  
#blobs 保存了所有的 blob， 顺序为从下到上从左到右  
def find_blobs (img_thre, ROIS, blob_h = 20, blob_w = 10):
    blobs = []
    blob  = []
    blob_start = 0
    blob_end   = 0
    blob_find  = False
    for roi in ROIS:
        roi_img = img_thre[roi[1]:(roi[1]+roi[3]),0:640]
        #cv.imshow('roi',roi_img)
        #cv.waitKey(0)
        roi_img_row = roi_img.sum(axis=0)
        for i in range(len(roi_img_row)):
            if blob_find : 
                if roi_img_row[i] > blob_h:
                    blob_end = i
                    if (blob_end - blob_start) > blob_w:
                        blob.append([blob_start, (blob_end - blob_start + 1), roi[1]])
                    blob_find = False
            else:
                if roi_img_row[i] < blob_h:
                    blob_find = True
                    blob_start = i
        if blob:
            blobs.append(blob)
        blob = []
    return blobs   #blob [start, length, ]

#返回中心
def center_blob(blob):
    center = [int(blob[2]+25), int(blob[0]+blob[1]/2)]
    return center

#确定两个blob之间的点是否为黑点，直线检测时的检测步骤
def check_block(center_new, center_old, img_thre, block_size = 4):
    center = [(center_old[0]-center_new[0]), (center_old[1]-center_new[1])]
    img_check = img_thre[(center_old[0]-center[0]//2-block_size//2):(center_old[0]-center[0]//2+block_size//2),
                        (center_old[1]-center[1]//2-block_size//2):(center_old[1]-center[1]//2+block_size//2)]
    
    #cv.imshow('img_check',img_check)
    #cv.waitKey(0)
    print (img_check.sum())
    m = ( img_check.sum() == 0  )
    return m

#找直线
def find_line (blobs,img_thre):
    line = []
    max_firstblob = 0
    for i in range(len(blobs)):
        if i == 0 :
            if len(blobs[0]) > 1:
                for j in blobs[0]:
                    if j[1] > max_firstblob:
                        max_firstblob = j[1]
                for j in blobs[0]:
                    if j[1] < max_firstblob:
                        blobs[0].remove(j)
                center_old = center_blob(blobs[0][0])
            else:
                center_old = center_blob(blobs[0][0])
            line.append(center_old)
        else:
            for j in blobs[i]:
                center_new = center_blob(j)
                if check_block(center_new,center_old,img_thre):
                    line.append(center_new)
                    center_old = center_new
    return line
    


while(True):
# get a frame   ver1.0
    _,img = cap.read()
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

#二值化操作，100以上设为255
    r1,img_thre = cv.threshold(img,100,255,cv.THRESH_BINARY)
    img_thre = img_thre / 255


    cv.imshow('img_thre',img_thre)
    '''
    cv.imshow('img_ini',img)
    cv.waitKey(0)'''

    blobs = find_blobs(img_thre, ROIS)
    if blobs:print (blobs)
    if blobs:
        for i in blobs:
                for j in i:
                    cv.rectangle(img,(j[0],j[2]), ((j[0]+j[1]),j[2]+ROI_h),(0,255,0),3)
    line = find_line(blobs, img_thre)
    print (line)
#ver 1.0
    if blobs:
        cv.line(img, (line[0][1],line[0][0]), (line[-1][1],line[-1][0]), (255,255,255), 10)
        #ver 2.2
        if len(blobs) == 1:
            for i in blobs:
                for j in i:
                    cv.line(img, (j[0],j[2]+25), ((j[0]+j[1]),j[2]+ROI_h-25), (255,255,255), 10)
    cv.imshow('Image',img)
#ver 1.0
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
