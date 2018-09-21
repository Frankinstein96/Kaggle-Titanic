# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:10:15 2018

@author: Mitali
"""
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Importing train data
train = pd.read_csv("titanic_train.csv")

#Visualising train data for null values
sns.heatmap(x_train.isnull())

#Splitting the train dataset
train.drop('Cabin',axis = 1,inplace = True)
x_train = train.drop('Survived',axis = 1)
y_train = train['Survived']

#Filling the missing values of Age column in train dataset with its mean.

from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values ="NaN",strategy = "mean",axis = 0)
imputer_train=imputer_train.fit(x_train['Age'].values.reshape(-1,1))
x_train['Age']=imputer_train.transform(x_train['Age'].values.reshape(-1,1))


#Importing test data
test = pd.read_csv('titanic_test.csv')
x_test = train.drop('Survived',axis = 1)
y_test = train['Survived']

#Visualising test data for null values
sns.heatmap(x_test.isnull())

#Filling the missing values of Age column in test dataset with its mean.
from sklearn.preprocessing import Imputer
imputer_test =  Imputer(missing_values ="NaN",strategy = "mean",axis = 0)
imputer_test = imputer_test.fit(x_test['Age'].values.reshape(-1,1))
x_test['Age'] = imputer_test.transform(x_test['Age'].values.reshape(-1,1))



# Converting catergorical to dummy variables in train data
Sex=pd.get_dummies(x_train['Sex'],drop_first=True)
Embark=pd.get_dummies(x_train['Embarked'],drop_first=True)
Pclass=pd.get_dummies(x_train['Pclass'],drop_first=True)

# Converting catergorical to dummy variables in test data
Sex=pd.get_dummies(x_test['Sex'],drop_first=True)
Embark=pd.get_dummies(x_test['Embarked'],drop_first=True)
Pclass=pd.get_dummies(x_test['Pclass'],drop_first=True)

#Merging the dummy variables to the train data
x_train = pd.concat([x_train,Sex,Embark,Pclass],axis=1)
x_test = pd.concat([x_test,Sex,Embark,Pclass],axis=1)

#Merging the dummy variables to the test data
x_train.drop(['PassengerId','Name','Ticket','Sex','Embarked','Pclass'],axis=1,inplace =True)
x_test.drop(['PassengerId','Name','Ticket','Sex','Embarked','Pclass'],axis=1,inplace =True)

#Training the dataset using Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

#Predicting for test dataset
predictions = logmodel.predict(x_test)

#Evaluating the model
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))





