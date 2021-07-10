import random
import numpy as np

def two_point_crossover(p1, p2):

    p1 = list(p1)
    p2 = list(p2)
    
    x1 = random.randint(1,len(p1)-2)
    x2 = random.randint(1,len(p1)-2)


    while x1 == x2:
        x2 = random.randint(1,len(p1)-2)
    
    if x2 < x1:
        x1, x2 = x2, x1
    print("x1:",x1)
    print("x2:",x2)
    return p1[:x1].copy() + p2[x1:x2].copy() + p1[x2:].copy()


p1 = np.random.randint(0,100,10)
p2 = np.random.randint(0,100,10)

print(p1)
print(p2)

print(two_point_crossover(p1,p2))