import numpy as np
import cv2
import matplotlib.pyplot as plt

# read video
cap = cv2.VideoCapture('D:\\projects-2022\\drowsiness\\newdataset\\YawDD dataset\\Dash\\Female\\1-FemaleNoGlasses.m4v')

ret, frame = cap.read()    
plt.figure()
plt.imshow(frame)
print(frame.shape)