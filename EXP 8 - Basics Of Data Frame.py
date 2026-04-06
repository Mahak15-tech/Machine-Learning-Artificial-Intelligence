#!/usr/bin/env python
# coding: utf-8

# Creating Series

# In[1]:


#Exp no.3


# In[2]:


#Aim: Basics of Data Frame


# In[3]:


# Name: Mahak Sunil Kamble
# Roll No: 24
# Sec: C
# Subject: ET-1
# Date: 22/08/2025


# In[4]:


# Program for creating Series using series() method of Pandas Library 


# In[5]:


#Import Pandas library
import pandas as pd 


# In[6]:


#Creating Student Name List
Name=["Ashish","Sagar","Shrikant","Lovly","Saurabh"]
Name


# In[7]:


#Creating a Series 
Roll_list=pd.Series(Name,index=[1,2,3,4,5])
print(Roll_list)


# Creating Data Frame

# In[8]:


#Import Pandas library
import pandas as pd 


# In[9]:


#Creating Data Frame
df=pd.DataFrame([[10,12,14,16],[11,18,15,21],[12,13,14,15]],
columns= ["CD","DBMS","DSS","CAO"])


# In[10]:


df


# In[11]:


df.shape


# In[12]:


df.size


# In[13]:


df.ndim


# Adding Record (Row) to the data frame

# In[14]:


df2=pd.DataFrame([[5,6,7,8]],
columns= ["CD","DBMS","DSS","CAO"])


# In[15]:


df3=df.append(df2,ignore_index=True)


# In[16]:


df3


# Adding Attribute (Columns) name DM

# In[17]:


df3["DM"]=[12,15,16,14]


# In[18]:


df3


# Deleting Record from df3 dataframe

# In[19]:


df4=df3.drop(index=[1])


# In[20]:


df4


# Deleting Attribute (Columns) from df3 dataframe

# In[21]:


df5=df3.drop(columns=["DM"])


# In[22]:


df5


# In[23]:


#Finding mean of DSS 

print("Mean of DSS:", df5["DSS"].mean())


# In[24]:


#Finding median of DSS

print("Median of DSS:", df5["DSS"].median())


# In[25]:


#Finding mode of DSS

print("Mode of DSS:", df5["DSS"].mode())


# In[26]:


#Finding max number of DSS

print("Max of DSS:", df5["DSS"].max())


# In[27]:


#Finding Min number of DSS

print("Max of DSS:", df5["DSS"].min())

