# -*- coding: utf-8 -*-
"""
Tue Apr  9 15:21:33 IST 2019

@author: Mohd Omama
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time


model=load_model(r'TrainedModels/Fire-64x64-color.model')

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


IMG_SIZE = 64
while(1):

    rval, image = vc.read()
    if rval==True:
        orig = image.copy()
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        tic = time.time()
        fire_prob = (1 - model.predict(image)[0][0]) * 100
        toc = time.time()
        print("Time taken = ", toc - tic)
        print("FPS: ", 1 / (toc - tic))
        print("Fire Probability: ", fire_prob)
        print(image.shape)
        
        label = "Fire Probability: " + str(fire_prob)
        cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        #fps=vc.get(cv2.CAP_PROP_FPS)
        #print(fps)
        
        cv2.imshow("Output", orig)
        
        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
    elif rval==False:
            break
end = time.time()


vc.release()
cv2.destroyWindow("preview")
