from __future__ import division
import csv
import math as m

rows=[]
filename="bank.csv"

with open(filename,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)
a=float(rows[0][0])

print(a)
print(a+a)
