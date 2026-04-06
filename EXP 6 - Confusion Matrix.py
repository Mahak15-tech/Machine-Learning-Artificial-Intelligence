#!/usr/bin/env python
# coding: utf-8

# # Finding parameters of Confusion Matrix

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


os.chdir('C:\\Users\\mahak\\OneDrive\\Desktop')


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


# # Splitting of DataSet into train and Test

# In[16]:


x=data.drop("target", axis=1)
y=data["target"]


# In[17]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# In[18]:


x_train


# In[19]:


x_test


# In[20]:


y_train


# In[21]:


y_test


# # Logistic Regression

# In[22]:


data.head()


# In[23]:


from sklearn.linear_model import LogisticRegression


# In[24]:


log = LogisticRegression()
log.fit(x_train, y_train)


# In[25]:


y_pred1=log.predict(x_test)


# In[26]:


from sklearn.metrics import accuracy_score 


# In[27]:


accuracy_score (y_test,y_pred1)


# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[29]:


cm = confusion_matrix(y_test, y_pred1)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

