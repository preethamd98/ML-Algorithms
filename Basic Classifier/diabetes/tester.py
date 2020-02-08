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

with open('diabetes.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for i in readCSV:
        data.append(i)


X = list()
Y = list()

data=np.delete(data,0,axis=0)
data=np.delete(data,(len(data)-1),axis=0)
np.random.shuffle(data);

for i in range(len(data)):
    temp = [1]
    for j in range(len(data[i])-1):
        temp.append((float(data[i][j])))
    X.append(temp)
    Y.append(data[i][len(data[0])-1])
    
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


