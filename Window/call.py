# -*- coding: utf-8 -*-\

import cv2 as cv
from Windowhandler import Windowhandler

ADDRESS = 'Image/4.jpg'


def main():
    img = cv.imread(ADDRESS, cv.IMREAD_COLOR)
    if img is None:
        print ('no input image')
    obj = Windowhandler(img)
    obj.WindowFinding()



if __name__ == '__main__':
    main()