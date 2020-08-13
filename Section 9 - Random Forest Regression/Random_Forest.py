#Random Fortest Regression

#Importing the llibreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#Fitting the Random Forest Regression to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(X,Y)

#predicting the New result
y_pred=regressor.predict(np.array([[6.5]]))

#Visualising the Random Forest regrssion result (For Higher Resolution and Smoother)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or bluff (Random Forest Regression)')
plt.xlabel('Positin Level')
plt.ylabel('Salary')
plt.show()