# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:35:53 2020

@author: Ishparsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data sets
dataset=pd.read_csv('Churn_Modelling.csv')

#importing datasets into dependant variable
X=dataset.iloc[:,3:13].values

#importing datasets into independent variables
Y=dataset.iloc[:,13].values

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X1=LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2=LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])

#onehotencoder=OneHotEncoder()
#X=onehotencoder.fit_transform(X).toarray()
#instead of onehotencoder we use column transformer
ct1=ColumnTransformer([('OneHot',OneHotEncoder(),[1])],remainder='passthrough')
X=ct1.fit_transform(X.tolist())
ct2=ColumnTransformer([('Onehot2',OneHotEncoder(),[2])],remainder='passthrough')
X=ct2.fit_transform(X.tolist())
X = X.astype('float64')
X=X[:,1:]

#spliting the datasets into training set and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.20, random_state=0)

#features Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


#making of ANN

#importing keras

import keras
from keras.optimizers import SGD
from keras.models import Sequential #initialize the model
from keras.layers import Dense #helps to build the layer


#initialize ANN
classifier=Sequential()

#adding input layers and first hidden layers
classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu' ,input_dim =12 ))
#adding second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))

#adding output layer
classifier.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))

#compiling ANN
#if accuracy does not vary the use this 
#opt = SGD(lr=0.01)
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])




#fitting the ANN into training sets
classifier.fit( X_train, Y_train,  batch_size=10, epochs=100)



#predicting the train values
#we need a classifier
y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)


#if we wanna input array randomly to see prediction then
new_predict=classifier.predict(np.array([[0,1,0,600,1,40,3,6000,2,1,1,50000]]))

#Making confusion matrix
#this is used to check the accuracy
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, y_pred)
