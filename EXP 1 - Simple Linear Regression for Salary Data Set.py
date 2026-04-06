#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Exp no.1


# In[2]:


#Aim: Simple Linear Regression for Salary Data Set


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[5]:


import os


# In[6]:


os.getcwd()


# In[7]:


os.chdir('C:\\Users\\mahak\\OneDrive\\Desktop')


# In[8]:


data=pd.read_csv("Salary_Data.csv")


# In[9]:


data.head()


# In[10]:


data.tail()


# In[11]:


data.info


# In[12]:


data.shape


# In[13]:


data.size


# In[14]:


data.ndim


# In[15]:


data.describe()


# In[16]:


data.isna()


# In[17]:


data.isna().any()


# In[18]:


data.isna().sum()


# INDEPENDENT AND DEPENDENT VARIABLES

# In[19]:


X=data.drop('Salary',axis=1)


# In[20]:


X.head()


# In[21]:


y=data.Salary


# In[22]:


y.head()


# In[23]:


from matplotlib import pyplot as plt


# In[24]:


plt.plot(X,y)
plt.title("Line Chart")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=50)


# In[26]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.30)


# In[27]:


print(X_train.shape)


# In[28]:


X_test.shape


# MODEL FITTING

# In[29]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,y_train)


# In[30]:


#Assigning Coefficient (slope) to m
m = LR.coef_


# In[31]:


print("Coefficient  :" , m)


# In[32]:


#Assigning Y-intercept to a
c = LR.intercept_


# In[33]:


print("Intercept : ", c)


# In[34]:


Accuracy = LR.score(X_test, y_test)
Accuracy


# y = mx+c

# PREDICTION

# In[35]:


y_pred=LR.predict(X_test)


# In[36]:


y_pred


# In[37]:


y_test


# EVALUATION METRICS

# In[38]:


from sklearn import metrics


# In[39]:


Accuracy = LR.score(X_test, y_test)
Accuracy


# MAE(MEAN ABSOLUTE ERROR)

# In[40]:


print(metrics.mean_absolute_error(y_test,y_pred))


# MSE(MEAN SQUARE ERROR)

# In[41]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# RMSE(ROOT MEAN SQUARE ERROR)

# In[42]:


rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print("RMSE:", rmse)

