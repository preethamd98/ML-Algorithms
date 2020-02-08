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
    temp = [0]*Y_size
    temp[Y_lookup[i]] = 1
    Y_data.append(temp)

training_p = len(X)*0.70
testing_p = len(X)*0.30
actual_training_p = training_p*0.80
validation_p = training_p*0.20

#np.random.shuffle(X)
X_training = X[:int(training_p)]
X_actual_training = X_training[:int(actual_training_p)]
X_validation = X_training[int(actual_training_p):]
X_testing = X[int(training_p):]

#np.random.shuffle(Y_data)
Y_training = Y_data[:int(training_p)]
Y_actual_training = Y_training[:int(actual_training_p)]
Y_validation = Y_training[int(actual_training_p):]
Y_testing = Y_data[int(training_p):]

# Have to write the lambda part
lmdam=[0,0.1,0.01,0.001,1,10,100,1000]

#lmda=random.randint(1,9)*np.identity(len(X[0]))
#lmda=np.array(lmda)

W = []

lp=np.zeros(len(X_actual_training[0]))
lp=np.array(lp);
X_actual_training=np.array(X_actual_training)

for i in X_actual_training:
    lp=lp+np.outer(i,i)

Wf=[]

for lmda in lmdam:    
    
    lmd=lmda*np.identity(len(X[0]))
    lp=lp+lmd

    lp = np.linalg.inv(lp)

    rp=np.zeros((len(X_actual_training[0]),len(Y_actual_training[0])))
    rp=np.array(rp)

    for i in range(len(X_actual_training)):
        rp=rp+np.outer(X_actual_training[i],Y_actual_training[i])
    
    W=np.matmul(lp,rp) 
    Wf.append(W);


Results = list()


for lmda in range(len(lmdam)): 
    temp_results = list()
    for i in range(len(X_validation)):
        temp = list()
        for k in range(len(W[0])):
               temp.append(np.dot(Wf[lmda][:,k],X_validation[i]))
        bit = [0]*len(Y_set)
        bit[temp.index(max(temp))] = 1
        temp_results.append(bit)
    Results.append(temp_results)
    
errors = list()
for lmda in range(len(lmdam)): 
    temp = 0
    for i in range(len(Results[lmda])):
        if(Results[lmda][i]==Y_validation[i]):
            temp+=1
    errors.append(temp)
    
Final_lambda_index = errors.index(max(errors))
Final_lambda = lmdam[errors.index(max(errors))]
Final_W = Wf[errors.index(max(errors))]   


Final_results = list()
for i in range(len(X_testing)):
   temp = list()
   for k in range(len(Final_W[0])):
       temp.append(np.dot(Final_W[:,k],X_testing[i]))

   bit = [0]*len(Y_set)
   bit[temp.index(max(temp))] = 1
   Final_results.append(bit)
      
temp = 0;
for i in range(len(Final_results)):
    if(Final_results[i]==Y_testing[i]):
        temp+=1
accuracy = temp/len(Y_testing)
print(accuracy)
    

cnf = list()
for i in range(Y_size):
    temp = [0]*Y_size
    cnf.append(temp)

test = Y[int(training_p):]



results = list()
for i in Final_results:
    for j in range(len(i)):
        if(i[j]==1):
            results.append(j)
    



for i in range(len(Y_testing)):
        cnf[test[i]-1][results[i]]+=1
        
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