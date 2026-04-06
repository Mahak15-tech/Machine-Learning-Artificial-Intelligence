#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the basic library
import pandas as pd 


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir("C:\\Users\\mahak\\OneDrive\\Desktop")


# In[5]:


data=pd.read_csv("diabetes.csv")


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


# In[13]:


data.columns


# In[14]:


data.isna()


# In[15]:


data.isna().any()


# In[16]:


data.isna().sum()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[18]:


#correlation
corr = data.corr()


# In[19]:


sns.heatmap(data.corr())


# In[20]:


plt.figure(figsize=(14,6))
sns.heatmap(data.corr())


# In[21]:


plt.figure(figsize=(14,6))
sns.heatmap(data.corr(),annot=True)

