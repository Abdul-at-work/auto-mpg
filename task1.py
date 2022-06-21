import pandas as pd
from math import sqrt
import statistics
from statistics import mode
# calculate the Euclidean distance between two vectors


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2)-2):
        if (row1[i]!=''):
            distance += (float(row1[i]) - row2[i])**2
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
    print(distances2)    
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

# iterating till the range
lst = []
print("Enter the value for each Category")
for i in range(0, 9):
    ele = input()

    lst.append(ele)  # adding the element

print(lst)
k = int(input("number of neighbours "))
car_data = pd.read_csv("auto-mpg.data", delim_whitespace=True, header=None)
df = pd.DataFrame(car_data)
dataset = df.values.tolist()
neighbors,distances = get_neighbors(dataset,lst, k)
for neighbor in neighbors:
    print(neighbor)

pnumber = int(input("Enter the column to predict "))
if (pnumber == 1 or pnumber == 7 or pnumber == 6):
    pvalue = categorypredict(neighbors, pnumber)
    print(pvalue)
else:
    pvalue = meanpredict(neighbors, pnumber,distances,k)
    print(pvalue)
