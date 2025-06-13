import cv2 as cv
import numpy as np

# Using bitwise operations we can basically perform masking, it is method of focusing on certain part of the image.
# take the example of photos of people and by applying mask on it you can focus on their faces.
#  size of mask should be same as the image

img = cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_2.png')
cv.imshow('image',img)

# blank image for masking 
blank= np.zeros(img.shape[:2],dtype = 'uint8')

mask= cv.circle (blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
cv.imshow('mask',mask)

masked= cv.bitwise_and(img,img,mask=mask)
cv.imshow("masked image",masked)

cv.waitKey(0)