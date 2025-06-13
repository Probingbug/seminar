import cv2 as cv
import numpy as np

blank= np.zeros((400,400),dtype = 'uint8')
img1= cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_2.png')
img2=cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Screenshots for testing/north_pole2.png')

rectangle= cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)

cv.imshow('rectangle',rectangle)
cv.imshow('circle',circle)

# Bitwise And

bitwise_and = cv.bitwise_and(img1, img2 )
cv.imshow("bitwise And ", bitwise_and)

# Bitwise OR

bitwise_or = cv.bitwise_or(img1,img2)
cv.imshow('bitwise OR',bitwise_or)

#  Bitwise XOR : non intersecting regions

bitwise_XOR = cv.bitwise_xor(img1,img2)
cv.imshow('bitwise XOR',bitwise_XOR)

#  bitwise NOT

bitwise_NOT = cv.bitwise_not(rectangle)
cv.imshow('bitwise NOT',bitwise_NOT)

# size of both the images should be same that's why it is not going to work with my objective


cv.waitKey(0)