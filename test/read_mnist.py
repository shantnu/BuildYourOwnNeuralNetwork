#!/usr/bin/env python
# coding: utf-8

# In[1]:


training_data_file = open("mnist_test.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[2]:


print(training_data_list[0])


# In[3]:


hand_digit = training_data_list[0][0]
raw_values = training_data_list[0][2:].split(',')



# In[4]:


import numpy as np
raw_values = np.asfarray(raw_values).reshape(28,28)


# In[5]:


# In[6]:


print("Value in CSV file: ", hand_digit)
