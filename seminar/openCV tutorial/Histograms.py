import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

 
img = cv.imread('/Users/anupam/mtech /assignments/2nd sem/seminar/seminar/selenographic image/north pole/north pole_1.png')
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("image",img)
# img_hist= cv.calcHist([gray ],[0],None,[256],[0,256])

# with mask

blank=np.zeros(img.shape[:2],dtype='uint8')


plt.figure()
plt.title('color Histogram')
plt.xlabel('Bins')
plt.ylabel('no of pixels')
# plt.plot(img_hist)
colors=('b','g','r')

for i, col in enumerate(colors):
    hist=cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])



# plt.xlim([0,256])
plt.show()

cv.waitKey(0)