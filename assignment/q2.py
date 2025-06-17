from numpy import random
import numpy as np
a=random.rand(20)*10
print(a)
b=np.round(a,2)
print(b)

import statistics
mean=statistics.mean(b)
median=statistics.median(b)
min=min(b)
maximum=max(b)
print(mean, median,maximum,min)

for j in range(0,20):
    if b[j]<=5:
        b[j]=b[j]*b[j]
        b=np.round(b,2)
print(b)

def alternate_sort(array):
    l=len(array)
    for i in range(0,(l-1)):
        for j in range(0,(l-1)):
             if array[j]>array[j+1]:
                 temp=array[j]
                 array[j]=array[j+1]
                 array[j+1]=temp
    return array
print(alternate_sort(b))
