# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:08:56 2020

@author: Rukon
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
import _tkinter
import tkinter
from tkinter import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

#tkinter._test()
"""
X_test = np.array([[1,1,3,2,0,0]]) #vgood


loaded_model = pickle.load(open("car_evaluation_model.sav", 'rb'))
res = loaded_model.predict(X_test)
#result = loaded_model.score(X_test, Y_test)
print(res)
"""

fields = ('Buying Rate', 'Maintenance', 'Doors', 'Persons', 'Luggage Boot Size', 'Safety', 'Result')
def car_condition(entries):
   buy = (float(entries['Buying Rate'].get()))
   maint = (float(entries['Maintenance'].get()))
   doors = (float(entries['Doors'].get()))
   persons = (float(entries['Persons'].get()))
   lug_boot = (float(entries['Luggage Boot Size'].get()))
   safety = (float(entries['Safety'].get()))
   
   #X_test = np.array([[3,3,0,0,2,1]]) #unacc
   X_test = np.array([[buy, maint, doors, persons, lug_boot, safety]])
   print(X_test)
  # sc = StandardScaler()
   #X_test = sc.fit_transform(X_test)
   print(X_test)
   loaded_model = pickle.load(open("car_evaluation_model.sav", 'rb'))
   res = loaded_model.predict(X_test)
   entries['Result'].delete(0, END)
   entries['Result'].insert(0, res)
   print( f"Car condition: {res}")
    
def makeform(root, fields):
   entries = {}
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=22, text=field+": ", anchor='w')
      ent = Entry(row)
      ent.insert(0,"0")
      row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
      lab.pack(side = LEFT)
      ent.pack(side = RIGHT, expand = YES, fill = X)
      entries[field] = ent
   return entries

if __name__ == '__main__':
   root = Tk()
   ents = makeform(root, fields)
   root.bind('<Return>', (lambda event, e = ents: fetch(e)))
   b2 = Button(root, text='Show Car Condition',
   command=(lambda e = ents: car_condition(e)))
   b2.pack(side = LEFT, padx = 5, pady = 5)
   b3 = Button(root, text = 'Quit', command = root.quit)
   b3.pack(side = LEFT, padx = 5, pady = 5)
   root.mainloop()