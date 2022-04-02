import os,keras,cv2 #cv2 requiered for selective search on images
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    #iou=area intersected / Area of union
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


import xml.etree.ElementTree as ET
mytree=ET.parse('tuberculosis-phonecamera/tuberculosis-phone-0001.xml')
root=mytree.getroot()
print(root[3].tag)
count=0
for x in root.iter('bndbox'):
    xmin=x.find('xmin').text
    xmax=x.find('xmax').text
    ymin=x.find('ymin').text
    ymax=x.find('ymax').text
    print(xmin,ymin,xmax,ymax)
    count+=1



import glob
import os.path as path

IMAGE_PATH='tuberculosis-phonecamera'
file_paths=glob.glob(path.join(IMAGE_PATH,'*.png'))

img_array=cv2.imread(file_paths)

for image in file_paths:
    print(image)


for filename in glob.iglob('tuberculosis-phonecamera/*.jpg'):
    img_array=cv2.imread(filename)
    plt.imshow(img_array)
    plt.show()
    break
