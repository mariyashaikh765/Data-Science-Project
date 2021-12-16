#!/usr/bin/env python
# coding: utf-8

# # *THE SPARKS FOUNDATION- Data Science and Business Analytics - Dec'21*

# # *Name: Shaikh Mariya*

# #  *Prediction Using Unspervised ML - KMeans* 

# # *Task 2 - From the given Iris Dataset, predict the optimum number of clusters and represent it visually.*

# # Importing the Essential Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings('ignore')


# # Importing the Dataset and Visualizing it

# In[2]:


iris=pd.read_csv('C:\\Users\\SHAIKH MARIYA\\Downloads\\Iris.csv')


# In[3]:


iris.head()


# In[4]:


iris.shape


# In[5]:


iris.describe()


# In[6]:


iris.info()


# # Cleaning the Dataset

# In[7]:


iris.drop(['Species', 'Id'], axis=1, inplace=True)
iris.head()


# # Examining the Distribution of Datapoints

# In[8]:


sns.scatterplot(x=iris['SepalLengthCm'], y=iris['SepalWidthCm'])
plt.title('Distribution of Points', color='r', size= 15)
plt.xlabel('Species', color='r', size=10)
plt.ylabel('Sepal Length (cm)', color='r', size=10)
plt.show()


# # *Application of Unsupervised Learning Model*

# In[9]:


model= KMeans(n_clusters=3)
model


# # Making Predictions

# In[10]:


prediction= model.fit_predict(iris)
prediction


# In[11]:


iris['cluster']=prediction
iris.head()


# # Segregating the Species to make Predictions

# In[12]:


sns.violinplot(x=iris['cluster'], y=iris['SepalLengthCm'])
plt.title('Segregating the Species', color='purple', size= 15)
plt.xlabel('Species', color='r', size=10)
plt.ylabel('SepalLengthCm', color='r', size=10)
plt.show()


# # Examining the Centroids

# In[13]:


centers=model.cluster_centers_
centers


# In[14]:


palette = ['tab:blue', 'tab:orange', 'tab:green']
sns.scatterplot(iris['SepalLengthCm'], iris['SepalWidthCm'], hue=iris['cluster'], palette=palette)
plt.scatter(centers[:, 0], centers[:,1], color='r', marker='*', s=150)
plt.title('Representing the Centroids', color='r', size= 15)
plt.xlabel('Sepal length (cm)', color='r', size=10)
plt.ylabel('Sepal width (cm)', color='r', size=10)
plt.show()


# In[15]:


k_model= range(1,10)
sse=[]
for k in k_model:
    model= KMeans(n_clusters=k)
    model.fit_predict(iris[['SepalLengthCm','SepalWidthCm']])
    sse.append(model.inertia_)


# In[16]:


plt.plot(k_model, sse, marker='o', markerfacecolor='r')


# # Conclusion: This Dataset has 3 Different Species, therefore it consists of 3 Clusters respectively

# # *Thank You*

# In[ ]:




