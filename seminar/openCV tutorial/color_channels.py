import cv2 as cv
import numpy as np

img= cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_4.png')
b,g,r = cv.split(img)

blank=np.zeros(img.shape[:2],dtype='uint8')


cv.imshow("blue",b)
cv.imshow('green',g)
cv.imshow('red',r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# merging will result the same image 

merged=cv.merge([b,g,r])


# instead of getting grey shade code for actual channel colors

blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])

cv.imshow("blue",blue)
cv.imshow('green',green)
cv.imshow('red',red)

cv.waitKey(0)



