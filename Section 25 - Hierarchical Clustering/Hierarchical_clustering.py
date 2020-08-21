# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:10:57 2019

@author: Kapil
"""
#Hierarchical Clustering

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Using the Dendrogram To Find Optimal no. of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidian Distance')
plt.show() 

#Fitting Hiererchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#Visualising the Clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,color='red',label='Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,color='blue',label='Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,color='green',label='Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,color='cyan',label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,color='magenta',label='Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()