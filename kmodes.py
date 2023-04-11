#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
bank = pd.read_csv(r"C:\dataset\bankmarketing.csv")


# In[2]:


bank.head()


# In[3]:


bank.columns


# In[4]:


# Importing Categorical Columns
bank_cust = bank[['age','job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','day_of_week','poutcome']]
bank_cust.head()


# In[5]:


# Converting age into categorical variable.
bank_cust['age_bin'] = pd.cut(bank_cust['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100],
 labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
bank_cust = bank_cust.drop('age',axis = 1)


# In[6]:


bank_cust.head()


# In[7]:


bank_cust.shape


# In[8]:


bank_cust.describe()


# In[9]:


bank_cust.info()


# In[10]:


# Checking Null values
bank_cust.isnull().sum()*100/bank_cust.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[11]:


#model building
# First we will keep a copy of data
bank_cust_copy = bank_cust.copy()
#Data Preparation
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bank_cust = bank_cust.apply(le.fit_transform)
bank_cust.head()


# In[12]:


# Importing Libraries
from kmodes.kmodes import KModes
#Using K-Mode with "Cao" initialization
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)


# In[13]:


# Predicted Clusters
fitClusters_cao


# In[14]:


clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = bank_cust.columns
# Mode of the clusters
clusterCentroidsDf


# In[15]:


#Using K-Mode with "Huang" initialization
km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(bank_cust)


# In[16]:


# Predicted clusters
fitClusters_huang


# In[17]:


#Choosing K by comparing Cost against each K
cost = []
for num_clusters in list(range(1,5)):
 kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
 kmode.fit_predict(bank_cust)
 cost.append(kmode.cost_)


# In[18]:


y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)


# In[19]:


## Choosing K=2
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)


# In[20]:


fitClusters_cao


# In[22]:


#Combining the predicted clusters with the original DF
bank_cust = bank_cust_copy.reset_index()
clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([bank_cust, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)
combinedDf.head()

