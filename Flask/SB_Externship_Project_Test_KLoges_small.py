# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:30:29 2021

@author: jayas
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:34:22 2021

@author: jayas
"""


from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf 

model=load_model(r"H:\My Drive\AI\Training\SmartBridge\Externship\Coding\Share Code\Jaya\cancer.h5")
img=image.load_img(r"C:\Users\Logu\Downloads\BreastCancer_Dataset_Small\Test\1\9023_idx5_x1301_y1401_class1.png", target_size=(128,128))
x=image.img_to_array(img)
print(x)
print(x.shape)

x=np.expand_dims(x, axis=0)
print(x.shape)

#pred=model.predict_classes(x)
y=model.predict(x)
pred= np.argmax(y, axis=1)
print(pred)

index=['Breast Cancer Negative','Breast Cancer Positive']
result=str(index[pred[0]])
result