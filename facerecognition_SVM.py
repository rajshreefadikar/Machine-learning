#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
 axi.imshow(faces.images[i], cmap='bone')
 axi.set(xticks=[], yticks=[],
 xlabel=faces.target_names[faces.target[i]])


# In[4]:


from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)


# In[6]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)


# In[7]:


from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
get_ipython().run_line_magic('time', 'grid.fit(Xtrain, ytrain)')
print(grid.best_params_)


# In[8]:


model = grid.best_estimator_
yfit = model.predict(Xtest)


# In[9]:


fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
 axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
 axi.set(xticks=[], yticks=[])
 axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
 color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# In[10]:


from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
 target_names=faces.target_names))

