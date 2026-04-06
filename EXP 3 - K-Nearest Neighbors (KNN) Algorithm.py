#!/usr/bin/env python
# coding: utf-8

# # To perform and analysis of K-Nearest Neighbors (KNN) Algorithm

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


# # Removing duplicates 

# In[16]:


data_dup =data.duplicated().any()


# In[17]:


data_dup


# In[18]:


data=data.drop_duplicates()


# In[19]:


data_dup =data.duplicated().any()


# In[20]:


data_dup


# # Splitting of DataSet into train and Test

# In[21]:


x=data.drop("target", axis=1)
y=data["target"]


# In[22]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# In[23]:


x_train


# In[24]:


x_test


# In[25]:


y_train


# In[26]:


y_test


# In[ ]:





# # KNN Classifier

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


# In[28]:


knn=KNeighborsClassifier()


# In[29]:


knn.fit(x_train, y_train)


# In[30]:


y_pred=knn.predict(x_test)


# In[31]:


accuracy = accuracy_score(y_test, y_pred)


# In[ ]:





# In[32]:


accuracy= accuracy_score (y_test,y_pred)


# In[33]:


score=[]
for k in range (1,10):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred3=knn.predict(x_test)
    score.append(accuracy_score (y_test,y_pred3)) 


# In[34]:


score


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred3)

labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# from above we can say that for value of K= 2 and 8 we can get maximum accuracy
