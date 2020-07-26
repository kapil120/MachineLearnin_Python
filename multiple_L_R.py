#Multiple Linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing The dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X=X[:,1:]

#Spliting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_Y=StandardScaler()
Y_train=sc_Y.fit_transform(Y_train)'''

#Fitting Multiple Linear Regression to the Trianing Set
from sklearn.linear_model import LinearRegression
regrssor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test Set Result
y_pred=regressor.predict(X_test)

#Building the Optimal model using Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones([50,1]).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()