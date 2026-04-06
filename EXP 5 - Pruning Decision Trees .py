#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[1]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas 

# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir("C:\\Users\\mahak\\OneDrive\\Desktop")


# In[5]:


data=pd.read_csv("heart.csv")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.shape


# In[11]:


data.size


# In[12]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment

# In[13]:


# check Missing Value by record 

data.isna()


# In[14]:


data.isna().any()


# In[15]:


data.isna().sum()


# # Independent and Dependent Variables
# 

# In[16]:


x=data.drop("target", axis=1)
y=data["target"]


# # Splitting of DataSet into train and Test¶

# In[17]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=60)


# # Decision Trees Algorithm

# In[18]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 


# In[19]:


dt=DecisionTreeClassifier()


# In[20]:


dt.fit(x_train, y_train)


# In[30]:


y_pred=dt.predict(x_test)


# In[31]:


accuracy_score (y_test,y_pred)


# In[32]:


print("Accuracy without pruning:", accuracy_score (y_test,y_pred))


# In[33]:


dt_depth = DecisionTreeClassifier(
    max_depth=12,
    random_state=42
)

dt_depth.fit(x_train, y_train)
y_pred_depth = dt_depth.predict(x_test)

print("Accuracy with max_depth pruning:", accuracy_score(y_test, y_pred_depth))


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt


# In[35]:


plt.figure(figsize=(20,10))
tree.plot_tree(
    dt,
    feature_names=x.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True
)
plt.show()


# In[36]:


plt.figure(figsize=(20,10))
tree.plot_tree(
    dt_depth,
    feature_names=x.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True
)
plt.show()


# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Train unpruned tree
clf_unpruned = DecisionTreeClassifier(random_state=42)
clf_unpruned.fit(x_train, y_train)

# Accuracy before pruning
y_pred = clf_unpruned.predict(x_test)
print("Accuracy before pruning:", accuracy_score(y_test, y_pred))

# Cost Complexity Pruning
path = clf_unpruned.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas

# Train pruned tree
clf_pruned = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
clf_pruned.fit(x_train, y_train)

y_pred_pruned = clf_pruned.predict(x_test)
print("Accuracy after pruning:", accuracy_score(y_test, y_pred_pruned))


# In[38]:


plt.figure(figsize=(20,10))
tree.plot_tree(
    clf_pruned ,
    feature_names=x.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True
)
plt.show()

