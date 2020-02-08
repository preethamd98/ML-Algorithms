#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:30:43 2019

@author: sanketh
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

with open('wine.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        data.append(i)


X = list()
Y = list()

data=np.delete(data,(len(data)-1),axis=0)
np.random.shuffle(data);

for i in range(len(data)-1):
    temp = [1]
    for j in range(1,len(data[i])-1):
        temp.append((float(data[i][j])))
    X.append(temp)
    Y.append(data[i][0])
    
Y_set = set(Y)
Y_lookup = dict()
count = 1


for i in range(len(Y)):
    Y[i] = int(Y[i])
    

for i in Y_set:
    Y_lookup[count] = count
    count+=1


Y_data = list()
Y_size = len(Y_set)

for i in Y:
    temp = [0]*Y_size
    temp[i-1] = 1
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
        
    
    





     
      

