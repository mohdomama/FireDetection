# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:41:43 2019

@author: Arpit Jadon
"""

from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time


model=load_model(r'MNetV2_Aug_NS_5Epoch.h5')
#cv2.namedWindow("preview")

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

fire_flag=False

start = time.time()
while(1):

    rval, image = vc.read()
    if rval==True:
        orig = image.copy()
        
        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        tic = time.time()
        (Fire, No_Fire)=model.predict(image)[0]
        toc = time.time()
        print("Time taken = ", toc - tic)
        print("FPS: ", 1 / (toc - tic))
        label = "Fire" if Fire > No_Fire else "No Fire"
        proba = Fire if Fire > No_Fire else No_Fire
        label = "{}: {:.2f}%".format(label, proba * 100)
        
        # draw the label on the image
        #output = imutils.resize(orig, width=400)
        output = cv2.resize(orig, (240,180))
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        #fps=vc.get(cv2.CAP_PROP_FPS)
        #print(fps)
        
        cv2.imshow("Output", output)
        
        
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break
    elif rval==False:
            break
end = time.time()

# Time elapsed
# seconds = end - start
# print("Time taken : {0} seconds".format(seconds))

# Calculate frames per second
# num_frames=90
# fps  = num_frames / seconds;
# print("Estimated frames per second : {0}".format(fps))

vc.release()
cv2.destroyWindow("preview")
