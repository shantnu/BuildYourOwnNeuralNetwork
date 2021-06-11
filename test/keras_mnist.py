#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop


# In[2]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


print(X_train.shape)
print(y_train.shape)
print(y_train[0])


# In[4]:


num_values = X_train.shape[0]

num_pixels = 784 # 28 * 28
X_train = X_train.reshape(num_values, num_pixels).astype('float32')


# In[5]:


num_values = X_test.shape[0]

num_pixels = 784 # 28 * 28
X_test = X_test.reshape(num_values, num_pixels).astype('float32')


# In[6]:


X_train = X_train / 255 * 0.99

X_test = X_test / 255 * 0.99


# In[7]:


np_utils.to_categorical(y_train)[0]


# In[8]:


print("Before: " , y_train[0])


# In[9]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

output_size = y_train[0].size


# In[10]:


print("After: " , y_train[0])


# In[11]:


model = Sequential()
model.add(Dense(200, input_dim=num_pixels, activation = 'relu', kernel_initializer='normal'))
model.add(Dense(output_size, activation = 'relu', kernel_initializer='normal'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),metrics=['accuracy'])


# In[12]:


print(len(y_train))


# In[13]:


model.fit(X_train, y_train, epochs = 2)


# In[14]:


score = model.evaluate(X_test, y_test)


# In[15]:


print(100 - score[1] * 100)
