# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:04:50 2022

@author: Sintia Ayu Listina
"""

import pandas as pd
from sklearn import neighbors
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_excel(r'D:\SEMESTER 8\SKRIPSI\DATA SKRIPSI/dataset_KNN_4param_juni-juli.xlsx')
data.head()
df = pd.DataFrame(data, columns=['HST','Kadar air tanah (%)','Suhu tanah (°C)','Relay (ml)','Kadar air tanah+1 (%)'])

train, test = train_test_split(df, test_size=0.2, random_state=69)

x_train = train.drop('Kadar air tanah+1 (%)', axis=1)
y_train = train['Kadar air tanah+1 (%)']

x_test = test.drop('Kadar air tanah+1 (%)', axis=1)
y_test = test['Kadar air tanah+1 (%)']
#print(x_test)
#print(y_test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

#import required packages
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


rmse_val = [] #to store rmse values for different k
acc_val = []
#k_value = range(1, 11)
#for K in k_value:
K = 5
model = neighbors.KNeighborsRegressor(n_neighbors = K)    
model.fit(x_train, y_train)  #fit the model
pred=model.predict(x_test) #make prediction on test set
error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
rmse_val.append(error) #store rmse values
print('RMSE value for k= ' , K , 'is:', error)
acc = (model.score(x_test, y_test))*100
acc_val.append(acc)
print(acc)

arr_result_uji = pred.reshape(41,1)

for n in range(len(arr_result_uji)):
    kat_value=format(arr_result_uji[n][0], '.2f')
    kat=float(kat_value)
    if kat < 34:
        print(kat,'Siram')
    else:
        print(kat,'Cukup')