import cv2 as cv
import numpy as np

img = cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_1.png')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)


# laplacian method

lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.


cv.waitKey(0)