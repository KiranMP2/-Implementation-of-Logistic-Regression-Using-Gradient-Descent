# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages

2.Read the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary and predict the Regression value
## Program:
```python
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KIRAN MP
RegisterNumber:  212224230123
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Placement_Data.csv")
df=df.drop("sl_no",axis=1)
df=df.drop("salary",axis=1)
df

```
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')

df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df

```
```python
print ("REGISTER NUMBER:212224230123")
print ("NAME:KIRAN MP")
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


theta = np.random.random(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))

def gradient_descent(theta, X,y, alpha, num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-= alpha*gradient
    return theta


theta = gradient_descent(theta,X,y,alpha = 0.01, num_iterations = 1000)

def predict(theta, X):
    h= sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)
y_pred
```
```python
accuracy = np.mean(y_pred.flatten()==y)
print("Name: KIRAN MP")
print("REG NO: 212224230123")
print("Accuracy",accuracy)

print(y_pred)

print(Y)

xnew= np.array([[0,87,0,95,0,2,0,0,1,0,0,0]])
y_prednew=predict(theta,xnew)
y_prednew
```
## Output:
<img width="1096" height="421" alt="Screenshot 2025-09-07 095115" src="https://github.com/user-attachments/assets/b4074f13-3762-4dc2-96f5-1d3ffde935c3" />
<img width="1011" height="451" alt="Screenshot 2025-09-07 095200" src="https://github.com/user-attachments/assets/fda45ef1-5971-4eff-93fd-6ccd6e01ae80" />
<img width="777" height="303" alt="Screenshot 2025-09-07 095454" src="https://github.com/user-attachments/assets/c61579d3-a3bb-4ba8-944d-260cde1a9b48" />
<img width="804" height="332" alt="Screenshot 2025-09-07 095607" src="https://github.com/user-attachments/assets/7ef0602b-d375-4f41-8448-b75e4890bb5e" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

