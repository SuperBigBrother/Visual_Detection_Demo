# -*- coding: utf-8 -*-
'''this code is used to change photoes to another size'''

import  cv2 as cv

ADDRESS_INI = 'data_initial/11.jpg'
ADDRESS_DES = 'data/11.jpg'

img = cv.imread(ADDRESS_INI,0)
img = cv.resize(img, (640,480))
cv.imwrite(ADDRESS_DES,img)
