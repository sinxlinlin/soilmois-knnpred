# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:37:54 2022

@author: Sintia Ayu Listina

5 parameter
"""

import pandas as pd
from sklearn import neighbors
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_excel(r'D:\SEMESTER 8\SKRIPSI\DATA SKRIPSI/dataset_imp_KNN_4param_juni-juli.xlsx')
data.head()
df = pd.DataFrame(data, columns=['HST','Kadar air tanah (%)','Suhu tanah (°C)','Relay (ml)','Kadar air tanah+1 (%)'])

train, test = train_test_split(df, test_size=0.2, random_state=60)

x_train = train.drop('Kadar air tanah+1 (%)', axis=1)
y_train = train['Kadar air tanah+1 (%)']

x_test = test.drop('Kadar air tanah+1 (%)', axis=1)
y_test = test['Kadar air tanah+1 (%)']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

#import required packages
from sklearn.metrics import mean_squared_error 
from math import sqrt

rmse_val = [] #to store rmse values for different k
#for K in range(20):
K = 7
model = neighbors.KNeighborsRegressor(n_neighbors = K)    
model.fit(x_train, y_train)  #fit the model
pred=model.predict(x_test) #make prediction on test set
error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
rmse_val.append(error) #store rmse values
print('RMSE value for k= ' , K , 'is:', error)
acc = model.score(x_test, y_test)
print(acc)

# save the model to disk
filename = 'finalskripsi_modelknn_4param_imp_rs60_k7.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(x_test, y_test)
#print(result)

# GUI python
import tkinter as tk
from tkinter import Label
from tkinter import LabelFrame
from tkinter import Button
from tkinter import Entry
from tkinter import Canvas


def resetpred():
    input01.delete(0, 'end')
    input02.delete(0, 'end')
    input03.delete(0, 'end')
    input04.delete(0, 'end')

def resultpred():
    a=float(input01.get())
    b=float(input02.get())
    c=float(input03.get())
    d=float(input04.get())
    X_baru = np.array([[a, b, c, d]])
    test = pd.DataFrame(X_baru)
    x_uji_scaled = scaler.transform(test)
    x_uji = pd.DataFrame(x_uji_scaled)
    prediction = model.predict(x_uji)
    result = format(prediction[0], '.2f')
    print(result)
    kat=float(result)
    dec_relay=[]
    if kat < 34:
        label11=Label(layar,text=('Lakukan Penyiraman'),fg="red",font=("Tahoma",9,"bold"), height=1, width=20)
        label11.place(x=292,y=330)
        print(kat,'Lakukan Penyiraman')
    else:
        label11=Label(layar,text=('Penyiraman Dicukupkan'),fg="blue",font=("Tahoma",9,"bold"), height=1, width=18)
        label11.place(x=310,y=330)
        print(kat,'Penyiraman Cukup')
    
    label01=Label(layar,text=(result,'%'),fg="green",font=("Tahoma",9,"bold"))
    label01.place(x=310,y=290)
    

layar=tk.Tk()
layar.title("Model Prediksi Kadar Air Tanah Untuk Keputusan Penyiraman")
layar.geometry("500x370")

canvas = Canvas()

label07=Label(layar,text="APLIKASI PREDIKSI KADAR AIR TANAH (%) UNTUK KEPUTUSAN PENYIRAMAN",fg="green",font=("Tahoma",9,"bold"))
label07.pack()
label08=Label(layar,text="KEBUN TOMAT BEEF DI GREENHOUSE SERENITY FARM",fg="green",font=("Tahoma",9,"bold"))
label08.pack()

label12=Label(layar,text="Masukan Data 3 Jam Sebelumnya:",fg="blue" ,font=("Tahoma",8,"bold"))
label12.place(x=15,y=50)

label02=Label(layar,text="Masukan HST Tanaman Tomat Beef",font=("Tahoma",7,"bold"))
label02.place(x=15,y=80)
input01=Entry()   
input01.place(x=18,y=100)

label03=Label(layar,text="Masukan Kadar Air Tanah (%)",font=("Tahoma",7,"bold"))
label03.place(x=275,y=80)
input02=Entry()   
input02.place(x=278,y=100)

label04=Label(layar,text="Masukan Suhu Tanah (°C)",font=("Tahoma",7,"bold"))
label04.place(x=15,y=140)
input03=Entry()   
input03.place(x=18,y=160)

label05=Label(layar,text="Masukan Jumlah Air Siram (mL/3 jam)",font=("Tahoma",7,"bold"))
label05.place(x=275,y=140)
input04=Entry()   
input04.place(x=278,y=160)


tombol01=Button(text="Prediksi",font=("Tahoma",8,"bold"),width=12,height=2,bg="yellow", command=resultpred)
tombol01.place(x=275,y=200)
tombol01=Button(text="Hapus",fg='white',font=("Tahoma",8,"bold"),width=12,height=2,bg="red", command=resetpred)
tombol01.place(x=385,y=200)

label06=Label(layar,text="Hasil Prediksi 3 Jam Berikutnya:",fg="blue" ,font=("Tahoma",8,"bold"))
label06.place(x=15,y=260)

label09=Label(layar,text="Kadar Air Tanah (%):",font=("Tahoma",8,"bold"))
label09.place(x=15,y=290)

label10=Label(layar,text="Keputusan Penyiraman:",font=("Tahoma",8,"bold"))
label10.place(x=15,y=330)
 
layar.mainloop()