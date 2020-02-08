#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:07:04 2019

@author: preethamdasari
"""

from __future__ import division
import numpy as np
import csv
import random
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


data = list()

with open('car.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        data.append(i)

#np.random.shuffle(data);

buying = list()  
maintainance = list()
doors = list()      
persons = list()     
lug_boot = list()     
safety = list()

condition = list()

np.random.shuffle(data);

for i in data:
    buying.append(i[0])
    maintainance.append(i[1])
    doors.append(i[2])
    persons.append(i[3])
    lug_boot.append(i[4])
    safety.append(i[5])
    condition.append(i[6])
    

buying_set = set(buying)  
maintainance_set = set(maintainance)  
doors_set = set(doors)  
persons_set = set(persons)  
lug_boot_set = set(lug_boot)  
safety_set = set(safety)  
condition_set = set(condition)

buying_lookup = dict()
maintainance_lookup = dict()
doors_lookup = dict()  
persons_lookup = dict()    
lug_boot_lookup = dict()    
safety_lookup = dict()   
condition_lookup = dict()

count = 1
for i in buying_set:
    buying_lookup[i] = count
    count+=1
    
count = 1
for i in maintainance_set:
    maintainance_lookup[i] = count
    count+=1

count = 1
for i in doors_set:
    doors_lookup[i] = count
    count+=1

count = 1
for i in persons_set:
    persons_lookup[i] = count
    count+=1
    
count = 1
for i in buying_set:
    buying_lookup[i] = count
    count+=1
    
count = 1
for i in lug_boot_set:
    lug_boot_lookup[i] = count
    count+=1
    
count = 1
for i in safety_set:
    safety_lookup[i] = count
    count+=1
    
count = 1
for i in condition_set:
    condition_lookup[i] = count
    count+=1
    
    
X = list()
Y = list()

for i in data:
    x = [1]
    x.append(buying_lookup[i[0]])
    x.append(maintainance_lookup[i[1]])
    x.append(doors_lookup[i[2]])
    x.append(persons_lookup[i[3]])
    x.append(lug_boot_lookup[i[4]])
    x.append(safety_lookup[i[5]])

    X.append(x)
    
for i in data:
    Y.append(condition_lookup[i[6]])
    
Y_set = set(Y)
Y_lookup = dict()
count = 0

for i in Y_set:
    Y_lookup[i] = count
    count+=1


Y_data = list()
Y_size = len(Y_set)

for i in Y:
    Y_data.append(Y_lookup[i])
    
training_p = len(X)*0.70
testing_p = len(X)*0.30

#np.random.shuffle(X)
X_training = X[:int(training_p)]
X_testing = X[int(training_p):]

#np.random.shuffle(Y_data)
Y_training = Y_data[:int(training_p)]
Y_testing = Y_data[int(training_p):]



## SVM

clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X_training,Y_training) 

Y_predict_svm = list()

for i in X_testing:
    Y_predict_svm.append(clf_svm.predict([i]))
    
temp = 0
for i in range(len(Y_predict_svm)):
    if(Y_predict_svm[i][0]==Y_testing[i]):
        temp+=1
    
accuracy = temp/len(Y_testing)
print("SVM Accuracy "+str(accuracy))


## Neural Networks

clf_ANN = MLPClassifier(solver='lbfgs', alpha=1e-5,random_state=1)
clf_ANN.fit(X_training,Y_training)

Y_predict_ann = list()

for i in X_testing:
    Y_predict_ann.append(clf_ANN.predict([i]))
    
temp = 0
for i in range(len(Y_predict_ann)):
    if(Y_predict_ann[i][0]==Y_testing[i]):
        temp+=1
    
accuracy_ann = temp/len(Y_testing)
print("ANN Accuracy "+str(accuracy_ann))

#Paste from here
cnf = list()
for i in range(Y_size):
    temp = [0]*Y_size
    cnf.append(temp)

for i in range(len(Y_testing)):
        cnf[Y_testing[i]][Y_predict_ann[i][0]]+=1
        
print(" ")

for i in cnf:
    print(i)
    
lab = list()
for i in Y_lookup:
    lab.append(i)    
    
    
df_cm = pd.DataFrame(cnf, index = [i for i in lab],
                  columns = [i for i in lab])
plt.figure(figsize = (10,7))
plt.title("Confusion Matrix")
sn.heatmap(df_cm, annot=True)

    



