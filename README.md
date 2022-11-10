# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries such as pandas module to read the corresponding csv file.
2. Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the corresponding dataset values.
4. Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.
5. Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
6. Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report
 
 


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANJAY S
RegisterNumber: 22007761
*/

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

![ml4op1](https://user-images.githubusercontent.com/115128955/201033680-901c5fea-ac2c-4fd7-a5b0-8ded891007e8.png)

![ml4op2](https://user-images.githubusercontent.com/115128955/201033752-94a28b73-705c-49a4-af31-7157792ddb23.png)

![ml4op3](https://user-images.githubusercontent.com/115128955/201033787-b86388a6-46ed-4532-9dba-c0223a24ffbd.png)

![ml4op4](https://user-images.githubusercontent.com/115128955/201033820-211192d0-a62d-43fd-aa31-19b829fcf85a.png)

![ml4op5](https://user-images.githubusercontent.com/115128955/201033860-518e2e07-3b3c-42d4-a695-01fc845cddab.png)

![ml4op6](https://user-images.githubusercontent.com/115128955/201033889-170efe41-78af-445c-acac-adb6e8716e03.png)

![ml4op7](https://user-images.githubusercontent.com/115128955/201033955-78849905-48be-4630-99d2-6c4689a8831a.png)

![ml4op8](https://user-images.githubusercontent.com/115128955/201034030-d4e74e97-ffe8-4c18-82dc-c5f4fd85a93d.png)

![ml4op9](https://user-images.githubusercontent.com/115128955/201034067-478095d9-8889-4116-8317-c887b7d36784.png)

![ml4op10](https://user-images.githubusercontent.com/115128955/201034154-198c277c-2423-4cde-9b6f-14a10e2bacab.png)

![ml4op11](https://user-images.githubusercontent.com/115128955/201034208-5f9446af-b324-4647-82f1-1ef1e1a5660a.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
