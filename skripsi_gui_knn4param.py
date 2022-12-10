# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:34:50 2022

@author: Sintia Ayu Listina
"""

import tkinter as tk
from tkinter import Label
from tkinter import Button
from tkinter import Entry

def resetpred():
    input01.delete(0, 'end')
    input02.delete(0, 'end')
    input03.delete(0, 'end')
    input04.delete(0, 'end')
    input05.delete(0, 'end')

def resultpred():
    a=float(input01.get())
    b=float(input02.get())
    c=float(input03.get())
    d=float(input04.get())
    e=float(input05.get())
    accmodel = model.score(x_test, y_test)
    X_baru = np.array([[a, b, c, d, e]])
    print(accmodel)
    test = pd.DataFrame(X_baru)
    prediction = model.predict(test)
    result = format(prediction[0], '.2f')
    print(result)
    
    label01=Label(layar,text=(result,'%'),fg="green",font=("Tahoma",9,"bold"))
    label01.place(x=290,y=220)
    

layar=tk.Tk()
layar.title("Model Prediksi Kadar Air Tanah 3 Jam Berikutnya")
layar.geometry("450x260")


label07=Label(layar,text="APLIKASI PREDIKSI KADAR AIR TANAH (%)",fg="green",font=("Tahoma",9,"bold"))
label07.pack(padx=5, pady=7, side=tk.TOP)


label02=Label(layar,text="Masukan Kelembaban Relatif Udara (%)",font=("Tahoma",7,"bold"))
label02.place(x=10,y=40)
input01=Entry()   
input01.place(x=13,y=60)

label03=Label(layar,text="Masukan Kadar Air Tanah (%)",font=("Tahoma",7,"bold"))
label03.place(x=230,y=40)
input02=Entry()   
input02.place(x=233,y=60)

label04=Label(layar,text="Masukan Suhu Tanah (°C)",font=("Tahoma",7,"bold"))
label04.place(x=10,y=100)
input03=Entry()   
input03.place(x=13,y=120)

label05=Label(layar,text="Masukan Jumlah Air Siram (mL/3 jam)",font=("Tahoma",7,"bold"))
label05.place(x=230,y=100)
input04=Entry()   
input04.place(x=233,y=120)

label06=Label(layar,text="Masukan HST Tanaman Tomat Beef",font=("Tahoma",7,"bold"))
label06.place(x=10,y=160)
input05=Entry()   
input05.place(x=13,y=180)

tombol01=Button(text="Prediksi",font=("Tahoma",8,"bold"),width=12,height=2,bg="yellow", command=resultpred)
tombol01.place(x=230,y=160)
tombol01=Button(text="Hapus",fg='white',font=("Tahoma",8,"bold"),width=12,height=2,bg="red", command=resetpred)
tombol01.place(x=335,y=160)

label06=Label(layar,text="Hasil Prediksi Kadar Air Tanah 3 Jam berikutnya:",font=("Tahoma",8,"bold"))
label06.place(x=10,y=220)
 
layar.mainloop()