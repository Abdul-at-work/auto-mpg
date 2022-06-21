import pandas as pd
from math import sqrt
import statistics
from statistics import mode
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import timeit

# calculate the Euclidean distance between two vectors







def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2)-2):
        distance += (float(row1[i]) - float(row2[i]))**2
    return sqrt(distance)

# Locate the most similar neighbors
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# iterating till the range
car_data = pd.read_csv("auto-mpg.data", delim_whitespace=True, header=None)
df = pd.DataFrame(car_data)
df.head()
df.drop(df.columns[8],axis=1,inplace=True)
y=df.iloc[:,0].values.tolist()
y=np.array(y)
y=y.astype(np.float64)
df.drop(df.columns[0],axis=1,inplace=True)
dataset = df.values.tolist()
dataset=np.array(dataset)
dataset=dataset.astype(np.float64)
pvalue=[]
tvalue=[]
pnumber = 0
kfold=KFold(n_splits=3)
for k in range (2,11):
    start = timeit.default_timer()
    for train,test in kfold.split(dataset):
        knn = KNeighborsRegressor(n_neighbors = k)
        X_train, X_test = dataset[train], dataset[test]
        y_train, y_test = y[train], y[test]
        knn.fit(X_train,y_train)
        pvalue=knn.predict(X_test)
        tvalue=y_test
    print("For K= ",k," MSE-Value: ",mean_squared_error(tvalue,pvalue,squared=True))
    print("For K= ",k," RMSE-Value: ",mean_squared_error(tvalue,pvalue,squared=False))
    print("For K= ",k," MAPE-Value: ",mean_absolute_percentage_error(tvalue,pvalue))
    stop = timeit.default_timer()
    print("For K= ",k, ' Time: ', stop - start)    

start = timeit.default_timer()
for train,test in kfold.split(dataset):
    tree = DecisionTreeRegressor()
    X_train, X_test = dataset[train], dataset[test]
    y_train, y_test = y[train], y[test]
    tree.fit(X_train,y_train)
    pvalue1=tree.predict(X_test)
    tvalue1=y_test
print("For Decision Regressor Tree MSE-Value: ",mean_squared_error(tvalue1,pvalue1,squared=True))
print("For Decision Regressor Tree RMSE-Value:",mean_squared_error(tvalue1,pvalue1,squared=False))
print("For Decision Regressor Tree MAPE-Value: ",mean_absolute_percentage_error(tvalue1,pvalue1))
stop = timeit.default_timer()
print(' Time: ', stop - start)