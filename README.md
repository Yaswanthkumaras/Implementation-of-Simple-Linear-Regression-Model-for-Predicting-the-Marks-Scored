# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Set variables for assigning dataset values.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rithika L
RegisterNumber: 212224230231 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print(df)

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
## DATASET
<img width="1410" height="573" alt="image" src="https://github.com/user-attachments/assets/bba1d918-e4f5-4648-9052-fede7d9be4b3" />

## HEAD VALUES
<img width="1383" height="125" alt="image" src="https://github.com/user-attachments/assets/1b6c93fa-fec2-435f-812f-a0fed6ba92f4" />

## TAIL VALUES
<img width="1351" height="123" alt="image" src="https://github.com/user-attachments/assets/afc31743-7b6f-47d4-bb44-52c0f5a90c9c" />
## X AND Y VALUES
<img width="1420" height="578" alt="image" src="https://github.com/user-attachments/assets/142f3363-e7d0-42d6-9a1e-dbb492f8751c" />
## PREDICTED VALUES OF X AND Y
<img width="1377" height="68" alt="image" src="https://github.com/user-attachments/assets/3e9d8f52-9d54-40d6-922e-3a5e2df73177" />
## TRAINING SET
<img width="1413" height="562" alt="image" src="https://github.com/user-attachments/assets/3fbe67ad-02a1-4e1e-9f7d-a629978fb2af" />
## TESTING SET AND MSE,MAE and RMSE
<img width="1406" height="705" alt="image" src="https://github.com/user-attachments/assets/ccb475b0-1ea0-4b2d-86fd-78c839626203" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
