#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
plt.scatter(x, y)
plt.show()


# In[2]:


from sklearn.cluster import KMeans
data = list(zip(x, y))
inertias = []
for i in range(1,11):
 kmeans = KMeans(n_clusters=i)
 kmeans.fit(data)
 inertias.append(kmeans.inertia_)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[3]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x, y, c=kmeans.labels_)
plt.show()


# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv(r"C:\dataset\selling_data.csv")
df.shape


# In[7]:


df.head()


# In[8]:


df.info() #View summary of dataset


# In[9]:


df.isnull().sum() #Check for missing values in dataset


# In[10]:


df.drop(['react_comment_r', 'react_share_r'], axis=1, inplace=True) #Drop null columns


# In[11]:


df.info() #Again view summary of dataset


# In[12]:


df.describe() #View the statistical summary of numerical variables


# In[13]:


# view the labels in the variable
df['status_id'].unique()


# In[14]:


# view how many different types of variables are there
len(df['status_id'].unique())


# In[15]:


# view the labels in the variable
df['status_published'].unique()


# In[16]:


# view how many different types of variables are there
len(df['status_published'].unique())


# In[17]:


# view the labels in the variable
df['status_type'].unique()


# In[18]:


# view how many different types of variables are there
len(df['status_type'].unique())


# In[19]:


df.drop(['status_id', 'status_published'], axis=1, inplace=True) 


# In[20]:


df.info() #View the summary of dataset again


# In[21]:


df.head() #Preview the dataset again


# In[22]:


#Declare feature vector and target variable
X = df
y = df['status_type']


# In[23]:


#Convert categorical variable into integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
y = le.transform(y)


# In[24]:


#Convert categorical variable into integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
y = le.transform(y)


# In[25]:


#View the summary of X
X.info()


# In[26]:


#Preview the dataset X
X.head()


# In[27]:


#Feature Scaling
cols = X.columns
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])
X.head()


# In[28]:


#K-Means model with two clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)


# In[29]:


#Check quality of weak classification by the model
labels = kmeans.labels_
# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[30]:


print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[31]:


#Use elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
 kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
 kmeans.fit(X)
 cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[32]:


#K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[33]:


#K-Means model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[34]:


#K-Means model with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X)
# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[ ]:




