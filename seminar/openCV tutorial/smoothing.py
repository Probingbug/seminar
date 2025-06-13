import cv2 as cv

img= cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/Equirectangular/Equi_2.png')
cv.imshow("image",img)

# by smoothing we try to reduce noise from the image, so we use blurring,

# concepts : 
'''
kernel : a small segment of image consist of some rows and columns, these rows and columns decide the size of kernel

blurring effect mainly work on the middle pixel and because of its surronding

types of blurring :

1. Averaging :
in this intensity of middle pixel gets decided by the average of intensity of the surrounding pixels.and this process 
happens overall in the image

image variable = cv.blur(image, kernel size)

2. gaussian blur :

here you give a perticular value to a perticular pixel then blurring is applied

format 
image variable = cv.GaussianBlur(img,kernal size,sigma_X,sigma_Y)


3. median blurring

instead of finding average it finds median of surrounding pixels
generally it is more effective in reducing the noise from the image as compare to the averaging method
used in removing some salt and papper noise in the image
used in advanced computer vision project that tend to reduce the substantial noise of image

format

variable = cv.medianBlur( source_image, kernal size in integer)

generally gaussian blur is not meant to be used with large size of kernals

Bilateral Blurring : blurring effect with retained the edges







'''


average = cv.blur(img,(3,3)) 
# (3,3) is the size of kernel, higher the kernel size more the blurring happens
cv.imshow('blurred Image',average)

gaussian =cv.GaussianBlur(img,(7,7),0)
cv.imshow('gaussian Blur',gaussian)

median= cv.medianBlur(img,3)
cv.imshow('median blur',median)

bilateral= cv.bilateralFilter(img,10,35,25)
cv.imshow('Bilateral blur',bilateral)



































cv.waitKey(0)