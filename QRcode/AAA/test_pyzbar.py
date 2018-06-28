import cv2 as cv
from pyzbar.pyzbar import decode

address = 'image/qr1.png'
img = cv.imread(address)
result = decode(img)

print (result)
print (result[0].data)

