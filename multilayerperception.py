#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# In[6]:


data = pd.read_csv(r"C:\dataset\iris.csv")


# In[7]:


data.sample(5)


# In[8]:


data.head(5)


# In[9]:


data.describe()


# In[10]:


# Normalize the data
df_norm = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)


# In[11]:


df_norm.describe()


# In[12]:


target = data[['variety']].replace(['Setosa','Versicolor','Virginica'],[0,1,2])
target.sample(n=5)


# In[13]:


df = pd.concat([df_norm, target], axis=1)
df.sample(n=5)


# In[14]:


train, test = train_test_split(df, test_size = 0.3)
trainX = train[['sepal.length','sepal.width','petal.length','petal.width']]# taking the training data features
trainY=train.variety# output of our training data
testX= test[['sepal.length','sepal.width','petal.length','petal.width']] # taking test data features
testY =test.variety #output value of test data
trainX.head(5)


# In[15]:


trainY.head(5)


# In[16]:


testX.head(5)


# In[17]:


# Solver is the weight optimizer: ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)


# In[18]:


clf.fit(trainX, trainY)


# In[19]:


prediction = clf.predict(testX)
print(prediction)


# In[20]:


print(testY.values)


# In[21]:


from sklearn import metrics
print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(prediction,testY))


# In[22]:


clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)


# In[23]:


clf.fit(trainX, trainY)


# In[24]:


prediction = clf.predict(testX)
print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(prediction,testY))

