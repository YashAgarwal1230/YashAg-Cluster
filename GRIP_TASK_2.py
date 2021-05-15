# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:18:30 2021

@author: Yash Agarwal
"""
"""GRIP TASK 1.ipynb

# GRIP @THE SPARKS FOUNDATION 
TASK 2 : Prediction using Unsupervised ML
AUTHOR : Yash Agarwal 
OBJECTIVE : From the given 'IRIS' dataset, predict the Optimum number of Clusters and represent it visually.
STEP 1 : Importing the Dataset from the given link.
"""

# import the required liberaries 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn import datasets
from sklearn.datasets import load_iris
df=pd.read_csv('C:\\Users\\Yash Agarwal\\Desktop\\Iris.csv',index_col=0,header=0)

df

"""STEP 2 : Visualizing our Data"""

df.head()

# Analysis
print(df.shape)
print(df.info())

print(df.isnull().sum())   # detects the missing values.
df.describe(include="all")  # generates the descriptive statistices which include the mean, median, mode.

y=df.values[:,:-1]
y

"""STEP 3: Find the optimum numnber of clusters.
"""

# We use the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

model = []
for i in range(5, 15):
    kmeans = KMeans(n_clusters = i, random_state = 10)
    kmeans.fit(y)
    model.append(kmeans.inertia_)
plt.plot(range(5, 15), model)
plt.scatter(range(5, 15),model)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Model')
plt.show()

print(model)

"""Apply the K-means Cluster on Data """

#fitting K-means to our dataset

kmeans = KMeans(n_clusters = 5, random_state = 10)
Y_pred =  kmeans.fit_predict(y)

#kmeans.fit(x)
#Y_pred =  kmeans.predict(x)

Y_pred

kmeans.n_iter_

df["predicted"]=Y_pred
df.head() #returns the first n rows of the dataset

df.predicted=df.predicted.map({0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'})
df.head()

"""STEP 5 : Visualizing the Cluster """

sns.lmplot(data=df,x="SepalLengthCm", y="SepalWidthCm",
             fit_reg=False, # No regression line
hue="predicted",palette="Set1")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
s = 200, c = 'blue')
plt.title('sepal length vs sepal width',size=20)
plt.show()
