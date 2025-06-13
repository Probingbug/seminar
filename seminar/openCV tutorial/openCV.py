import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# grayscaling a image

image= cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_2.png')


# img =  cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_4.png')
# # cv.imshow('north pole moon',img)
# # cv.waitKey(0)

# def rescaleFrame(frame,scale= 0.75):
#     # work with videos,images and live videos
#     width = int (frame.shape[1]*scale)
#     height= int (frame.shape[0]*scale)

#     dimensions= (width,height)
    
#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

# resized_image= rescaleFrame(img)

# # cv.imshow('image',resized_image)
# # cv.waitKey(0)


# # creating a dummy image to work upon

# blank = np.zeros((500,500,3),dtype='uint8')
# #  coloring the image green
# # blank[ : ]=240,2,240
# # cv.imshow('blank',blank)
# # cv.waitKey(0)

# # coloring a range of pixels

# # blank[200:300,300:500]=0,0,255
# # cv.imshow('blank',blank)
# # cv.waitKey(0)

# cv.rectangle(blank,(0,0),(250,250),(0,250,0),thickness=2)
# cv.imshow("rectangle",blank)
# # cv.waitKey(0)

# # filled rectangle

# cv.rectangle(blank,(0,0),(250,250),(0,250,0),thickness=cv.FILLED)
# cv.imshow("rectangle",blank)
# # cv.waitKey(0)

# # draw a circle

# cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,255),thickness=-1)
# cv.imshow('circle',blank)
# # cv.waitKey(0)

# # put text on image

# cv.putText(blank,'hello',(225,225),cv.FONT_HERSHEY_COMPLEX_SMALL,1.0,(0,255,0),2)
# cv.imshow('text image',blank)
# cv.waitKey(0)

# resized=cv.resize(image,(500,500),interpolation=cv.INTER_AREA)

# cropped_img=image[20:200,20:400]
# cv.imshow('original',image)
# cv.imshow('cropped',cropped_img)


# threshing the image

# ret,thresh= cv.threshold(image,80,200,cv.THRESH_BINARY)

# cv.imshow('threshold image',thresh)
# cv.waitKey(0)

# color scaling 

# BGR to RGB

rgb= cv.cvtColor(image,cv.COLOR_BGR2RGB)
# cv.imshow("RGB",rgb)
# cv.waitKey(0)



plt.imshow(rgb)
plt.show()

# splitting among color channels
