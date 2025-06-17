## Problem 1 
## i)
from numpy import random
a=random.randint(1,50,size=(5,4))
print(a)

## ii)
for i in range(0,5):
    b=max(a[i])
    print(b)

## iii)
sum=0
for i in a:
   for j in i:
        sum=sum+j
mean=sum/20
print(mean)
import numpy as np
arr=[]

for i in a:
     for j in i:
         if j<=mean:
            arr.append(j)
arr=np.array(arr)
print(arr)

## iv)

def boundary_traverse(matrix):
    rows=len(matrix)
    col=len(matrix[0])
    for i in range(rows):
        for j in range(col):
            if i==0 or i==2 or j==0 or j==3:
                print (matrix[i][j])

   
    return

matrix=[[1,20,34,43],[29,35,54,67],[51,52,37,53]]
boundary_traverse(matrix)