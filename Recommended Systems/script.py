from __future__ import division
from datetime import datetime
from sklearn import tree
import csv
import math
import matplotlib.pyplot as plt





k = [i for i in range(1,50)]
Threshold_Rating = 3

def PR_RE(TOP_PREDICTED,TOP_ACTUAL,TOP_MOVIE,TOTAL_RELEVENT,K):
    NEW_TOP_PREDICTED=list()
    NEW_TOP_ACTUAL=list()
    NEW_TOP_MOVIE=list()
    try:
        for f in range(K):
            NEW_TOP_PREDICTED.append(TOP_PREDICTED[f])
            NEW_TOP_ACTUAL.append(TOP_ACTUAL[f])
            NEW_TOP_MOVIE.append(TOP_MOVIE[f])
        Relevant = list()
        RECOMENDED = list()
        for j in range(K):
            if(NEW_TOP_ACTUAL[j]>=Threshold_Rating):
                Relevant.append(TOP_MOVIE[j])
            if(NEW_TOP_PREDICTED[j]>=Threshold_Rating):
                RECOMENDED.append(TOP_MOVIE[j])
        Precision = percent(len((set(Relevant)).intersection(set(RECOMENDED))),len(RECOMENDED))
        Recall = percent(len((set(Relevant)).intersection(set(RECOMENDED))),len(TOTAL_RELEVENT))
        PR.append(Precision)
        RE.append(Recall)
    except:
        AAA = 0
    avg_pr=sum(PR)/len(PR)
    avg_re=sum(RE)/len(RE)
    return (avg_pr,avg_re)


"""
        avg_pr=sum(PR)/len(PR)
        avg_re=sum(RE)/len(RE)
        Avg_pr.append(avg_pr)
        Avg_re.append(avg_re)

        NEW_TOP_PREDICTED=list()
        NEW_TOP_ACTUAL=list()
        NEW_TOP_MOVIE=list()
        try:
            for f in range(K):
                NEW_TOP_PREDICTED.append(TOP_PREDICTED[f])
                NEW_TOP_ACTUAL.append(TOP_ACTUAL[f])
                NEW_TOP_MOVIE.append(TOP_MOVIE[f])


            Relevant = list()
            RECOMENDED = list()
            for j in range(K):
                if(NEW_TOP_ACTUAL[j]>=Threshold_Rating):
                    Relevant.append(TOP_MOVIE[j])
                if(NEW_TOP_PREDICTED[j]>=Threshold_Rating):
                    RECOMENDED.append(TOP_MOVIE[j])
            Precision = percent(len((set(Relevant)).intersection(set(RECOMENDED))),len(RECOMENDED))
            Recall = percent(len((set(Relevant)).intersection(set(RECOMENDED))),len(TOTAL_RELEVENT))
            PR.append(Precision)
            RE.append(Recall)
        except:
            AAA = 0
    avg_pr=sum(PR)/len(PR)
    avg_re=sum(RE)/len(RE)
    Avg_pr.append(avg_pr)
    Avg_re.append(avg_re)
"""




















def split_data(A):
    B=A[:int(math.ceil(0.7*len(A)))]
    C=A[-(len(A)-int(math.ceil(0.7*len(A)))):]
    return B,C

def percent(A,B):
    if B==0:
        return 0
    else:
        return ((A/B)*100)


def order_sort(A,B,C):
    for i in range(len(A)):
        for j in range(len(A)):
            if(A[i]>A[j]):
                temp = A[i]
                A[i] = A[j]
                A[j] = temp
                temp2 = B[i]
                B[i] = B[j]
                B[j] = temp2
                temp3 = C[i]
                C[i] = C[j]
                C[j] =temp3
    return A,B,C


def accuracy(A,B):
    acc = 0
    for i in range(len(A)):
        if(A[i]==B[i]):
            acc+=1
    acc = acc*100/len(A)
    return acc


data_file = "data.csv"
item_file = "item.csv"
rows=list()
rows2=list()
with open(data_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)

with open(item_file,'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows2.append(row)



#Convert to integer
for i in rows:
    for j in i:
        j = int(j)

item_data_lookup=dict()

for i in rows2:
    i[0]=int(i[0])
    item_data_lookup[i[0]]=i[-19:]
    for j in range(len(item_data_lookup[i[0]])):
        item_data_lookup[i[0]][j]=int(item_data_lookup[i[0]][j])





user_id = list()
item_id = list()
rating = list()
time_stamp = list()


for i in rows:
    user_id.append(i[0])
    item_id.append(i[1])
    rating.append(i[2])
    time_stamp.append(i[3])


Users = list(set(user_id))
NUM_OF_USERS = len(Users)
user_rating_dict = dict()
user_rating_list = list()

for i in Users:
    temp = dict()
    for j in range(len(user_id)):
        if(i==user_id[j]):
            temp[int(item_id[j])]=int(rating[j])
    user_rating_dict[i]=temp
print("Finish Reading files")
now=datetime.now()
print(now.strftime("%X"))

PREDICTOR=list()
PREDICTOR.append(tree.DecisionTreeClassifier())
PREDICTOR=PREDICTOR*NUM_OF_USERS
PR = list()
RE = list()



for i in range(len(Users)):
    A=user_rating_dict[Users[i]]
    movie_id=list()
    temp_rating=list()
    temp_genre=list()
    for j in A:
        movie_id.append(j)
        temp_rating.append(A[j])
        temp_genre.append(item_data_lookup[j])
        train_movie_id,test_movie_id = split_data(movie_id)
        train_genre,test_genre = split_data(temp_genre)
        train_rating,test_rating = split_data(temp_rating)
        PREDICTOR[i].fit(train_genre,train_rating)
        ACTUAL = test_rating
        PREDICTED = PREDICTOR[i].predict(test_genre)
        TOP_PREDICTED,TOP_ACTUAL,TOP_MOVIE = order_sort(PREDICTED,ACTUAL,test_movie_id)
        TOTAL_RELEVENT = list()
    for j in TOP_ACTUAL:
        if(j>=Threshold_Rating):
            TOTAL_RELEVENT.append(j)
print("Finish Building Trees")
now=datetime.now()
print(now.strftime("%X"))

PR_RE_VALUES=list()
for i in k:
    PR_RE_VALUES.append(PR_RE(TOP_PREDICTED,TOP_ACTUAL,TOP_MOVIE,TOTAL_RELEVENT,i))

Avg_pr = list()
Avg_re = list()

for i in PR_RE_VALUES:
    Avg_pr.append(i[0])
    Avg_re.append(i[1])








del(rows)
print("Im done!")
now=datetime.now()
print(now.strftime("%X"))

plt.plot(k,Avg_pr)
plt.show()
plt.plot(k,Avg_re)
plt.show()
