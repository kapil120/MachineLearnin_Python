# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 00:04:39 2019

@author: Kapil
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#Part-1 Data PreProcessing

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values
 
#Encoding categorica data
#Encoding Independent Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1=LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2=LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])
onehotE=OneHotEncoder(categorical_features=[1])
X=onehotE.fit_transform(X).toarray()
X=X[:,1:]

#Splliting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train ,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Part-2 Let's Make the ANN

#Importing the Keras Libraries and Packages
import os
os.environ['keras_BACKEND']='theano'
import keras
import theano
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier=Sequential()

#Adding the Input layer and first  Hidden Layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#Adding the second Hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#Adding Output Layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train,Y_train,batch_size=10,epochs=100)

#Part 3 - Making the predictions and Evaluating the Model

# Predicting  the Test set Result
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Making The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
