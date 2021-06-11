#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from keras.applications.vgg16 import VGG16

from matplotlib import pyplot
from PIL import Image


# In[2]:


model = ResNet50(weights='imagenet')


# 1st Picture: Elephant
#
# ![](elephant.jpg)

# In[3]:


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


# Dog
#
# ![](European_Dobermann.jpg)

# In[4]:


img_path = 'European_Dobermann.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


# Cars
#
# ![](cars.jpg)

# In[5]:


img_path = 'cars.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


# Karate
#
# ![](karate.jpg)

# In[6]:


img_path = 'karate.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


# Guitar
#
# ![](guitar.png)

# In[7]:


img_path = 'guitar.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


# In[ ]:
