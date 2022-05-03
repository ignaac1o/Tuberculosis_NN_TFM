
import cv2
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("anotations/tuberculosis-phone-0001.csv")
img=cv2.imread("photos/tuberculosis-phone-0001.jpg")


for coordinate in df.iterrows():
    cv2.rectangle(img, (int(coordinate[1][0].split(" ")[0]), int(coordinate[1][0].split(" ")[1])), (int(coordinate[1][0].split(" ")[2]), int(coordinate[1][0].split(" ")[3])), (0, 255, 0), 1, cv2.LINE_AA)
    
    
plt.imshow(img)
plt.show()
   