import numpy as np
import math
import os.path
import csv
import glob
import tensorflow as tf
import h5py as h5py
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import imageio
import matplotlib.pyplot as plt
from mlxtend.image import extract_face_landmarks


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance.euclidean(eye[0], eye[1])
    p += distance.euclidean(eye[1], eye[2])
    p += distance.euclidean(eye[2], eye[3])
    p += distance.euclidean(eye[3], eye[4])
    p += distance.euclidean(eye[4], eye[5])
    p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area /(p**2)

def mouth_over_eye(eye):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def getFrame(sec):
    start = 180000
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start + sec*1000)
    hasFrames,image = vidcap.read()
    return hasFrames, image          	

#main program
from scipy.spatial import distance
import cv2
data = []
labels = []
for i in range(1,13): # 3 4 5
   #file='D:\\projects-2022\\drowsiness\\Fold5_part2\\' + str(i) + '.mov'
   file='D:\\projects-2022\\drowsiness\\newdataset\\YawDD dataset\\Dash\\Female\\' + str(i) + '-FemaleNoGlasses.avi'
   vidcap = cv2.VideoCapture(file)
   print(file)
   sec = 1
   frameRate = 1
   start = 1
   #vidcap.set(cv2.CAP_PROP_POS_MSEC, start + sec*1000)
   success, image = vidcap.read()
   #print(success, image)
   #plt.imshow(image)
   count = 0
   while success and count < 10: # 240
      
        landmarks = extract_face_landmarks(image)
        print(landmarks)
        if sum(sum(landmarks)) != 0:
            count += 1
            data.append(landmarks)
            labels.append([i])
            sec = sec + frameRate
            sec = round(sec, 2)
            success, image = getFrame(sec)
            #print(count)
        else:
            sec = sec + frameRate
            sec = round(sec, 2)
            success, image = getFrame(sec)
            print('not detected')
        
import numpy as np
data = np.array(data)
labels = np.array(labels)

features = []
for d in data:
  eye = d[38:68]
  
  ear = eye_aspect_ratio(eye)
  mar = mouth_aspect_ratio(eye)
  cir = circularity(eye)
  mouth_eye = mouth_over_eye(eye)

  features.append([ear, mar, cir, mouth_eye])

features = np.array(features)
features.shape


np.save(open('Data.npy', 'wb'),data) # write binary data
np.save(open('Features.npy', 'wb'),features)
np.save(open('Labels.npy', 'wb'),labels)
np.savetxt("Features.csv", features, delimiter = ",") # write text data
np.savetxt("Labels.csv", labels, delimiter = ",")

# a = np.load('Data.npy')
# b = np.load('Features.npy')
# c = np.load('Labels.npy')

print('Hello')