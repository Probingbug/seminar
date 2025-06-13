import cv2 as cv
import numpy as np

img= cv.imread("/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_1.png")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

threshold,thresh=cv.threshold(gray,120,255,cv.THRESH_BINARY)
cv.imshow('simple thresh',thresh)

# threshold,thresh_inv=cv.threshold(gray,90 ,255,cv.THRESH_BINARY_INV)
# cv.imshow('simple thresh inverse',thresh_inv)

adaptive_thresh= cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,2)
cv.imshow("adaptive thresh",adaptive_thresh)

cv.waitKey(0)