#Polynomial lineaer Regression

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#splitting the deteset in training and test set
'''from sklearn.model_selection import train_test_split
X_train,X_test ,Y_train ,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)'''

#Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visulaising The Linear Regrssion Results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth Or Bluff (Linear Regression) ')
plt.xlabel('Position Level')
plt.ylabel('Salary')

#Visualising The Polynomial Regression Results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth Or Bluff (Polynomial Regression) ')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a nw result with Linear Regression
lin_reg.predict(6.5)
#Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))