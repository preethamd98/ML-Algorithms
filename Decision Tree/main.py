from __future__ import division
import csv
import math as m
import numpy as np
from random import seed
from random import randrange

def evaluate_algorithm(data, decision_tree, k_folds, *args):
    scores = list()
    folds = list()
    if k_folds==1:
        folds=data
        train_set = data
        test_set = list()
        for row in folds:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[len(row_copy)-1] = None
        predicted = decision_tree(train_set, test_set, *args)
        actual = [row[len(row)-1] for row in folds]
        accuracy_p = accuracy(actual, predicted)
        scores.append(accuracy_p)

    else:
        folds = cross_validation_split(data, k_folds)
        for fold in folds:
		    train_set = list(folds)
		    train_set.remove(fold)
		    train_set = sum(train_set, [])
		    test_set = list()
		    for row in fold:
			   row_copy = list(row)
			   test_set.append(row_copy)
			   row_copy[len(row_copy)-1] = None
		    predicted = decision_tree(train_set, test_set, *args)
		    actual = [row[len(row)-1] for row in fold]
		    accuracy_p = accuracy(actual, predicted)
        scores.append(accuracy_p)
    return scores

def accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def cross_validation_split(data, k_folds):
	data_split = list()
	data_copy = list(data)
	fold_size = int(len(data) / k_folds)
	for i in range(k_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)
	return data_split

def terminal(group):
	outcomes = [row[len(row)-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size_set, depth):
	left, right = node['groups']
	del(node['groups'])

	if not left or not right:
		node['left'] = node['right'] = terminal(left + right)
		return

	if depth >= max_depth:
		node['left'], node['right'] = terminal(left), terminal(right)
		return

	if len(left) <= min_size_set:
		node['left'] = terminal(left)
	else:
		node['left'] = get_split_data(left)
		split(node['left'], max_depth, min_size_set, depth+1)

	if len(right) <= min_size_set:
		node['right'] = terminal(right)
	else:
		node['right'] = get_split_data(right)
		split(node['right'], max_depth, min_size_set, depth+1)

def build_tree(train, max_depth, min_size_set):
	root = get_split_data(train)
	split(root, max_depth, min_size_set, 1)
	return root

def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def decision_tree(train, test, max_depth, min_size_set):
	tree = build_tree(train, max_depth, min_size_set)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)

	return(predictions)


def test_split(rows,f_index,value):
    left=list()
    right=list()
    for i in range(len(rows)):
        if(rows[i][f_index]<value):
            left.append(rows[i])
        else:
            right.append(rows[i])
    return left,right

def entropy_of_spliting(groups,classes,number_of_samples):
    entropy = 0
    for i in groups:
        gp_size = len(i)/number_of_samples
        gp_sum = 0
        class_set = list()
        for k in i:
            class_set.append(k[len(k)-1])

        for j in classes:
            #gp_sum-=(class_set.count(j)/len(class_set))*m.log((class_set.count(j)/len(class_set)),2)
            try:
                gp_sum-=(class_set.count(j)/len(class_set))*m.log((class_set.count(j)/len(class_set)),2)
            except:
                gp_sum-=0

        entropy+=gp_size*gp_sum
    return entropy

def get_split_data(data):
    entropy = 0

    number_of_samples = len(data)
    class_set = list()
    for i in data:
        for j in i:
            class_set.append(j[len(j)-1])
    classes = list(set(class_set))


    for i in classes:
        entropy-=(class_set.count(i)/len(class_set))*m.log((class_set.count(i)/len(class_set)),2)



    best_information_gain = 0
    best_f_index = 0
    best_value = 0
    best_groups = 0

    for f_index in range(len(data[0])-1):
        f_index_values = list()
        for k in range(len(data)):
            f_index_values.append(data[k][f_index])
        for f_index_value in f_index_values:
            groups = test_split(data,f_index,f_index_value)
            entropy_of_split = entropy_of_spliting(groups,classes,number_of_samples)
            information_gain = entropy-entropy_of_split
            if(information_gain>best_information_gain):
                best_information_gain=information_gain
                best_f_index = f_index
                best_value = f_index_value
                best_groups = groups
    return {'gain':best_information_gain,'index':best_f_index,'value':best_value,'groups':best_groups}

filename = "bank.csv"
granularity = 0;

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

k_folds = 2
max_depth = 5
min_size_set = 10

MAX = [i for i in range(3,8)]
scor=list()
A=list()
acc=list()
for i in MAX:
    A=evaluate_algorithm(rows, decision_tree, k_folds, i, min_size_set)
    acc.append(sum(A)/float(len(A)))
print(acc)


"""
scores = evaluate_algorithm(rows, decision_tree, k_folds, max_depth, min_size_set)
#print(tree)
print('Scores: %s' % scores)
print('Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
"""
