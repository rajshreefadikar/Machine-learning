#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import scikit-learn dataset library
from sklearn import datasets
#Load dataset
cancer = datasets.load_breast_cancer()


# In[2]:


# print the names of the 13 features
print("Features: ", cancer.feature_names)
# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)


# In[3]:


# print data(feature)shape
cancer.data.shape


# In[4]:


# print the cancer data features (top 5 records)
print(cancer.data[0:5])


# In[5]:


# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)


# In[6]:


# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test


# In[7]:


#Import svm model
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel #kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[8]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[9]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[10]:


#tuning hyperparameters
#Create a svm Classifier
clf = svm.SVC(kernel='poly', degree = 2, gamma = 'scale') # Polynomial Kernel #gamma : {'scale', 'auto'}
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[11]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[13]:


df = pd.read_csv(r"C:\dataset\apples_and_oranges.csv")
df


# In[14]:


df.dtypes


# In[15]:


plt.xlabel('weight')
plt.ylabel('size')
df.plot.scatter(x='Weight', y='Size')
#plt.scatter(df['Weight'], df['Weight'],color="green",marker='+', linewidth='5')
#plt.scatter(df['Size'], df['Size'],color="blue",marker='.' , linewidth='5')


# In[16]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2)


# In[17]:


x_train = train_set.iloc[:,0:2].values
y_train = train_set.iloc[:,2].values
x_test = test_set.iloc[:,0:2].values
y_test = test_set.iloc[:,2].values


# In[18]:


len (x_train)


# In[19]:


len (x_test)


# In[20]:


from sklearn.svm import SVC
model = SVC(kernel='rbf', random_state = 1)
model.fit(x_train, y_train)


# In[21]:


model.score(x_test, y_test)


# In[22]:


model.predict([[55,4]])


# In[ ]:




