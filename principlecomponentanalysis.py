#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\dataset\iris.csv")


# In[3]:


from sklearn.preprocessing import StandardScaler
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['variety']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[4]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
 , columns = ['principal component 1', 'principal component 2'])


# In[5]:


principalDf


# In[6]:


finalDf = pd.concat([principalDf, df[['variety']]], axis = 1)
finalDf


# In[7]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Setosa', 'Versicolor', 'Virginica']
colors = ['r', 'g', 'b']
for variety, color in zip(targets,colors):
 indicesToKeep = finalDf['variety'] == variety
 ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
 , finalDf.loc[indicesToKeep, 'principal component 2']
 , c = color
 , s = 50)
ax.legend(variety)
ax.grid()


# In[8]:


pca.explained_variance_ratio_


# In[9]:


#using PCA and logistic regression
#DOWNLOAD AND LOAD THE DATA
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')


# In[10]:


#SPLIT DATA INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# In[11]:


#STANDARDIZE THE DATA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


# In[12]:


#IMPORT AND APPLY PCA
from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.95)
pca.fit(train_img)


# In[13]:


#APPLY THE MAPPING (TRANSFORM) TO THE TRAINING SET AND THE TEST SET.
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


# In[14]:


#APPLY LOGISTIC REGRESSION TO THE TRANSFORMED DATA
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)
logisticRegr.predict(test_img[0].reshape(1,-1))
logisticRegr.predict(test_img[0:10])


# In[15]:


#MEASURING MODEL PERFORMANCE
logisticRegr.score(test_img, test_lbl)


# In[ ]:




