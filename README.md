# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2. Upload the dataset and check any null value using .isnull() function.
3. Declare the default values for linear regression.
4. Calculate the loss usinng Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using scatter plot function. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thenmozhi p
RegisterNumber: 212221230116 
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
dh.head()

df.tail()

#segregating data to variables
x = df.iloc[:,:-1].values
x

y = df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual value
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="magenta")
plt.plot(x_train,regressor.predict(x_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="gray")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse = np.sqrt(mse)
print('RMSE',rmse)

```
### Output:
## df.head():

![image](https://user-images.githubusercontent.com/95198708/234175620-6e41e59f-bcf4-40fa-a0b5-5be8ac2ff6ec.png)

## df.tail():

![image](https://user-images.githubusercontent.com/95198708/234175675-2371bfee-e731-43bd-9cb6-b98d0b35da69.png)

## x values:

![image](https://user-images.githubusercontent.com/95198708/234175720-8da2d4a5-1adf-4b42-a95e-0f67a5ff87f1.png)

## y values:

![image](https://user-images.githubusercontent.com/95198708/234175775-6cc459b9-32eb-4d40-aa2e-e972f2f0ea55.png)

## y_pred:

![image](https://user-images.githubusercontent.com/95198708/234175830-b22fb7f4-8316-432d-a6e4-4f1b527307d5.png)

## y_test:

![image](https://user-images.githubusercontent.com/95198708/234175866-44bc98f5-e4e2-4e86-bc39-a4b0e5c10d32.png)

## Graph of training data:

![image](https://user-images.githubusercontent.com/95198708/234175905-9cbf29f3-8104-4c6d-88a9-ee843440daf9.png)

## Graph of test data:

![image](https://user-images.githubusercontent.com/95198708/234175973-bb058dc3-dd3b-4be6-aebc-53814a746953.png)


## Values of MSE, MAE, RMSE:

![image](https://user-images.githubusercontent.com/95198708/234175437-52a6a249-7a9e-4526-b07d-073779578b66.png)



## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
