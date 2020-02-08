import math
import csv
import numpy as np
import random

def distance(Z,v):
    D_ika_temp = np.matmul(((np.array(Z[k]))-v[i]),A)
    D_ika = np.matmul(D_ika_temp,((np.array(Z[k]))-v[i]).T)
    return D_ika

def obj_function(z,u,v,c,m):
    s = 0
    for i in range(c):
        for k in range(len(z)):
            s = s+(u[i][k]**m)*distance(z[k],v[i])
    return s

#Importing Data
filename = "Cluster.csv"
Z = list()
with open(filename,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        Z.append(row)
for i in Z:
    i[0]=float(i[0])
    i[1]=float(i[1]) 
    
mining = list()
classification = list()

#Dividing the given data into training and classification sets
for i in range(len(Z)):
    if (i+1)%5 == 0:
        classification.append(Z[i])
    else:
        mining.append(Z[i])

#Parameters
N = len(mining)    
m = 2
e = 0.01
J = [0,0]
#c = 4
A = np.eye(2)
iterations = list()


for c in range(2,12):
    #Initializing the partition matrix
    U_trans = list()
    U = list()
    for i in range(len(mining)):
       a = [random.randint(1,100) for j in range(c)]
       s=0
       for j in a:
           s=s+j
       a = [i/s for i in a]
       U_trans.append(a)
    U = np.transpose(U_trans) 
    
    l=0
    while True:
        U_curr = np.copy(U)
        
        #Centroids for clusters
        v = list()
        for i in range(c):
            n = [0,0]
            d = 0
            for k in range(N):
                p = np.multiply(U[i][k]**m,mining[k])
                n = np.add(n,p)
                d = d + U[i][k]**m
            v.append(list(np.divide(n,d)))
        
        #Calculating distances of all the points from centroids
        Skip = list()
        for i in range(c):
            for k in range(N):
                D_ika_temp = np.matmul(((np.array(Z[k]))-v[i]),A)
                D_ika = np.matmul(D_ika_temp,((np.array(Z[k]))-v[i]).T)
                if(k not in Skip):
                    
                    #Updating the partition matrix 
                    if(D_ika > 0):
                        sum = 0
                        for j in range(c):
                            D_jka_temp = np.matmul(np.subtract(Z[k],v[j]),A)
                            D_jka = np.matmul(D_jka_temp,((np.array(Z[k]))-v[j]).T)
                            sum+= math.pow((D_ika/D_jka),(2/(m-1)))
                        U[i][k] = 1/sum
                    else:
                        U[i][k] = 1
                        tt = [i for i in range(c)]
                        tt.remove(i)
                        Skip.append(k)
                        for l in tt:
                            U[l][k] = 0
        l+=1
        #Termination condition
        t = U_curr-U
        if(((abs(t)).max())<e):
            break
    
    iterations.append(l)
    J_ik = 0
    for i in range(c):
        for k in range(N):
            J_ik1 = np.matmul(((np.array(Z[k]))-v[i]),A)
            J_ik2 = np.matmul(J_ik1,((np.array(Z[k]))-v[i]).T)
            J_ik += J_ik2*U[i][k]
    J.append(J_ik)

R = [0,0,0]
for c in range(3,10):
    R_c = abs((J[c]-J[c+1])/(J[c-1]-J[c]))            
    R.append(R_c)

R_new = list()
for i in R:
    if i != 0:
        R_new.append(i)

C = 3 + R_new.index(min(R_new))

#C = 4

#Initializing the partition matrix
U_trans = list()
U = list()
for i in range(len(mining)):
   a = [random.randint(1,100) for j in range(C)]
   s=0
   for j in a:
       s=s+j
   a = [i/s for i in a]
   U_trans.append(a)
U = np.transpose(U_trans) 

l=0
while True:
    U_curr = np.copy(U)
    
    #Centroids for clusters
    v = list()
    for i in range(C):
        n = [0,0]
        d = 0
        for k in range(N):
            p = np.multiply(U[i][k]**m,mining[k])
            n = np.add(n,p)
            d = d + U[i][k]**m
        v.append(list(np.divide(n,d)))
    
    #Calculating distances of all the points from centroids
    Skip = list()
    for i in range(C):
        for k in range(N):
            D_ika_temp = np.matmul(((np.array(Z[k]))-v[i]),A)
            D_ika = np.matmul(D_ika_temp,((np.array(Z[k]))-v[i]).T)
            if(k not in Skip):
                
                #Updating the partition matrix 
                if(D_ika > 0):
                    sum = 0
                    for j in range(C):
                        D_jka_temp = np.matmul(np.subtract(Z[k],v[j]),A)
                        D_jka = np.matmul(D_jka_temp,((np.array(Z[k]))-v[j]).T)
                        sum+= math.pow((D_ika/D_jka),(2/(m-1)))
                    U[i][k] = 1/sum
                else:
                    U[i][k] = 1
                    tt = [i for i in range(C)]
                    tt.remove(i)
                    Skip.append(k)
                    for l in tt:
                        U[l][k] = 0
    l+=1
    #Termination condition
    t = U_curr-U
    if(((abs(t)).max())<e):
        break
for i in range(len(U)):
    U[i]=list(U[i])
 
U_transpose = np.transpose(U)
cluster=list()
for i in U_transpose:
    i = list(i)
    x=max(i)
    y=i.index(x)
    cluster.append(1+y)
#    
#final = np.copy(mining)
#for i in range(len(final)):
#    final[i].append(indices[i])
    
X_coor=[mining[i][0] for i in range(len(mining))]
Y_coor=[mining[i][1] for i in range(len(mining))]
