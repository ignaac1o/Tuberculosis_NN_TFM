
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv("DATA_SPLIT/val/anotations/tuberculosis-phone-0111.csv")
img=cv2.imread("data/tuberculosis-phone-1011.jpg")

img=np.array(img)
print(img.shape)

#for coordinate in df.iterrows():
#    cv2.rectangle(img, (int(coordinate[1][0].split(" ")[0]), int(coordinate[1][0].split(" ")[1])), (int(coordinate[1][0].split(" ")[2]), int(coordinate[1][0].split(" ")[3])), (0, 255, 0), 3, cv2.LINE_AA)
    
    
#plt.imshow(img)
#plt.show()


# Histogram view
#read image in grey scale



#img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#hist,bin = np.histogram(img.ravel(),256,[0,255])


#img=cv2.equalizeHist(img)

#plt.subplot(1,2,1)
#plt.imshow(img,cmap='gray')
#plt.title('Image: tuberculosis-phone-1261')
#plt.xticks([])
#plt.yticks([])

#plt.subplot(1,2,2)
#plt.hist(img.ravel(),256,[0,255])
#plt.title('histogram')

#plt.show()

