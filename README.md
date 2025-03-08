# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: thenmozhi.p
RegisterNumber:  212223100059
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
# dataset
![image](https://github.com/user-attachments/assets/e7c23ebe-8678-49ff-97d0-dafbe4735c60)
# head value and tail value
![image](https://github.com/user-attachments/assets/dbf1e98b-32ce-4f67-afac-a6b7632842f4)
# x and y value
![image](https://github.com/user-attachments/assets/6bf890b5-43a5-4a0c-bd26-b8ff1bbd7c24)
# Predication values of X and Y
![image](https://github.com/user-attachments/assets/414be4aa-970f-4209-85a8-4b162956f226)
# MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/fe412a07-176d-47eb-b353-f2012fe76444)
# Training Set

![image](https://github.com/user-attachments/assets/a4cce6a6-d446-47f7-b0f2-2254647271b8)
# Testing Set
![image](https://github.com/user-attachments/assets/8d45264d-02c9-4e3c-b48c-82a28897f661)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
