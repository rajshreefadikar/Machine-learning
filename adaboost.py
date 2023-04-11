#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# In[2]:


# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[3]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# In[4]:


# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
 learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)


# In[5]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[6]:


#using diff base learners
# Load libraries
from sklearn.ensemble import AdaBoostClassifier
# Import Support Vector Classifier
from sklearn.svm import SVC
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
svc=SVC(probability=True, kernel='linear')
# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[7]:


from sklearn.ensemble import RandomForestClassifier
# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(random_state = 101),
 learning_rate=0.01,random_state = 96)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[8]:


from sklearn.tree import DecisionTreeClassifier
# Creating a decision tree classifier instance
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)
# Create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=20, base_estimator=dtree,learning_rate=0.005,random_state = 96)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

