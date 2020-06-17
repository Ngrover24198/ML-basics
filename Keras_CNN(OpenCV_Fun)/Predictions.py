#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image


# In[2]:


model = tf.keras.models.load_model("mnist_opencv.h5")


# In[3]:


def pred_image(img):
    
    IMAGE_LENGTH = 28
    img = np.reshape(img,(IMAGE_LENGTH,IMAGE_LENGTH,1))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    pred_class = model.predict_classes(img_tensor)[0]
    
    print(pred_class)
    return pred_class


# In[4]:


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


# In[5]:


cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, img = cap.read()
    img, contours, thresh = get_img_contour_thresh(img)
    ans = 0
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            # print(predict(w_from_model,b_from_model,contour))
            x, y, w, h = cv2.boundingRect(contour)
            # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28,28))
#            newImage = np.array(newImage)
#            newImage = newImage.flatten()
#            newImage = newImage.reshape(newImage.shape[0], 1)
            ans = pred_image(newImage)

    x, y, w, h = 0, 0, 300, 300
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Deep Network :  " + str(ans), (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    k = cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

