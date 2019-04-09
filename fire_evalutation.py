# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:26:37 2019

@author: Admin
"""


# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:16:10 2019

@author: Arpit Jadon
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix 
import itertools
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Input
from keras.applications.mobilenetv2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping





test_data_dir1=  r'Datasets/new_test'

#For Testing on BoWFire Dataset
test_batches= ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_data_dir1, target_size=(64,64),batch_size=39, shuffle=False)

model=load_model('TrainedModels/Fire-64x64-color-v2.model')
test_labels=test_batches.classes
#test_labels
#test_batches.class_indices
predictions=model.predict_generator(test_batches, steps=381, verbose=1)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
test_batches.class_indices
cm_plot_labels=['Fire','NoFire']
# plot_confusion_matrix(cm, cm_plot_labels,title='Confusion Matrix')


# In[15]:

tp=cm[0]
tp1=tp[0]
tn=tp[1]
#print(tp1)
#print(tn)
Recall=tp/(tp+tn)
Recall1=(max(Recall))
fn=cm[1]
fn1=fn[0]
#print(fn1)
Precision=tp1/(tp1+fn1)
f1_measure= 2*((Precision*Recall)/(Precision+Recall))
f_measure=(max(f1_measure))

print("Precision="+str(Precision))
print("Recall="+str(Recall1))
print("f_measure="+str(f_measure))
