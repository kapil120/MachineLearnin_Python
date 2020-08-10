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
'''from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)'''

#Fitting the Regression model to the Dataset
#Create Your Regressor Here

#Predicting a new result
Y_pred=regressor.predict(6.5)



#Visualising The Polynomial Regression Results
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X)),color='blue')
plt.title('Truth Or Bluff (Regression model) ')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising The Polynomial Regression Results (For Higher Resolution and Smoother)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid)),color='blue')
plt.title('Truth Or Bluff (Regression model) ')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()