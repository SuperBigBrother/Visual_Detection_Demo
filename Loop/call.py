# -*- coding: utf-8 -*-\

import cv2 as cv
from Loophandler import Loophandler

ADDRESS = 'Image/1.jpg'


def main():
    img = cv.imread(ADDRESS, cv.IMREAD_COLOR)
    if img is None:
        print ('no input image')
    obj = Loophandler(img)
    obj.LoopFinding()



if __name__ == '__main__':
    main()