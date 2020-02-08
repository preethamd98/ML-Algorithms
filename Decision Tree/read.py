from __future__ import division
import csv
import math as m

def test_split(rows,f_index,value):
    left=list()
    right=list()
    for i in range(len(rows)):
        if(rows[i][f_index]<value):
            left.append(rows[i][f_index])
        else:
            right.append(rows[i][f_index])
    return left,right
"""
def entropy_of_split(groups,classes,number_of_samples):
    entropy = 0
    for i in group:
        gp_size = len(i)/number_of_samples
        gp_sum = 0
        for j in classes:
            A = dict()
            gp_class = list(set(i))
            for k in gp_class:
                A[k] = i.count(k)
            gp_sum-=(j/A[j])*m.log(j/A[j],2)
            entropy+=gp_size*gp_sum
    return entropy

def get_best_split():
    entropy = 0
    A=dict()
    B=list(set(data[len(data)-1]))
    for i in B:
        A[i] = data[len(data)-1].count(i)
    for i in class:
        entropy-=(A[i]/number_of_samples)*m.log(A[i]/number_of_samples,2)
    best_information_gain = 0
    for i in range(len(data)):
        for j in data[i]:
            groups = test_split(rows,i,m.mean(data[i]))
            entropy_of_split = entropy_of_split(groups,classes,number_of_samples)
            information_gain = entropy-entropy_of_split
            if(information_gain>best_information_gain):
                best_information_gain=information_gain
                best_f_index =i
                best_value = m.mean(data[i])
                best_groups = best_groups
    return best_information_gain,best_f_index,best_value,best_groups
"""





filename = "bank.csv"
granularity = 0;
fields = []
rows = []
number_of_attributes = 5
data_set_size = 0



with open(filename,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)


#Conversion into float
for i in rows:
    for j in i:
        j = float(j)



data = []

for i in range(len(rows[0])):
    col = list()
    for j in range(len(rows)):
        col.append(rows[j][i])
    data.append(col)
#print(len(set(data[4])))

data_set_size = len(data[0])
number_of_classes = len(list(set(data[number_of_attributes-1]))

#data 5*1372 rows 1372*5
