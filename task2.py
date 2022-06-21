import pandas as pd
from math import sqrt
import statistics
from statistics import mode
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error
import timeit

# calculate the Euclidean distance between two vectors


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2)-1):
        distance += (float(row1[i]) - float(row2[i]))**2
    return sqrt(distance)

# Locate the most similar neighbors


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    distances2=[]
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
        distances2.append(distances[i][1])
    return neighbors,distances2


def categorypredict(neighbors, pnumber):
    lst2 = []
    for rows in neighbors:
        lst2.append(rows[pnumber])
    return mode(lst2)

def meanpredict(neighbors, pnumber, distances,k):
    weight=[]
    sum=0
    sum2=0
    for d in distances:
        weight.append(1/d**2)
        sum+=1/d**2
    for i in range(k):
        sum2+=neighbors[i][pnumber]*weight[i]
    finalsum=sum2/sum
    return finalsum  

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# iterating till the range
car_data = pd.read_csv("auto-mpg.data", delim_whitespace=True, header=None)
df = pd.DataFrame(car_data)
df.head()
df.drop(df.columns[8],axis=1,inplace=True)
dataset = df.values.tolist()
dataset=np.array(dataset)
dataset=dataset.astype(np.float64)
print(dataset)
pvalue=[]
tvalue=[]
pnumber = 0
kfold=KFold(n_splits=3)
for k in range (2,11):
    start = timeit.default_timer()
    for train,test in kfold.split(dataset):
        for data in dataset[test]:
            neighbors,distances = get_neighbors(dataset[train],data, k)
            pvalue.append(meanpredict(neighbors, pnumber,distances,k))
            tvalue.append(data[pnumber])
    print("For K= ",k," MSE-Value: ",mean_squared_error(tvalue,pvalue,squared=True))
    print("For K= ",k," RMSE-Value: ",mean_squared_error(tvalue,pvalue,squared=False))        
    print("For K= ",k," MAPE-Value: ",mean_absolute_percentage_error(tvalue,pvalue))
    stop = timeit.default_timer()
    print("For K= ",k, ' Time: ', stop - start)