import cv2 as cv
import numpy as np


# grayscaling a image

image= cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_2.png')
gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow("gray image",gray)

# blurring an image

blur=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)

cv.imshow('blur image',blur)
# cv.waitKey(0)

# contour detection on gray blur image

contours,heirarchy=cv.findContours(blur,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f"{len(contours)} contours  on blur image found !")

# printing those contours using a blank image of same dimension

blank=np.zeros(image.shape,dtype='uint8')

cv.drawContours(blank,contours,-1,(0,0,255),1)
cv.imshow('drawn contours on blur image',blank)

# printing those contours using a blank image of same dimension

blank=np.zeros(image.shape,dtype='uint8')

cv.drawContours(blank,contours,-1,(0,0,255),1)
cv.imshow('drawn contours on gray image',blank)



# edge cascading

canny= cv.Canny(blur,125,175)
cv.imshow('canny',canny)
# cv.waitKey(0)

# contour detection on gray scaled image

contours,heirarchy=cv.findContours(gray,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f"{len(contours)} contours  on gray image found !")

# printing those contours using a blank image of same dimension

blank=np.zeros(image.shape,dtype='uint8')

cv.drawContours(blank,contours,-1,(0,0,255),1)
cv.imshow('drawn contours on gray image',blank)


# contour detection on canny image
contours,heirarchy=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f"{len(contours)} contours on canny image found !")

blank=np.zeros(image.shape,dtype='uint8')

cv.drawContours(blank,contours,-1,(0,0,255),1)
cv.imshow('drawn contours on canny image',blank)


# color spaces

# BGR to HSV

hsv= cv.cvtColor(image,cv.COLOR_BGR2HSV);
cv.imshow("HSV",hsv)


# printing those contours using a blank image of same dimension

blank=np.zeros(image.shape,dtype='uint8')

cv.drawContours(blank,contours,-1,(255,0,0),1)
cv.imshow('drawn contours on gray image',blank)


# contour detection on canny image
contours,heirarchy=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f"{len(contours)} contours on HSV image found !")

blank=np.zeros(image.shape,dtype='uint8')

cv.drawContours(blank,contours,-1,(255,0,0),1)
cv.imshow('drawn contours on HSV image',blank)




cv.waitKey(0)
