from svm_info import RandomForestClass
from utils import Utils
from individual import Individual
import numpy as np
import time
import multiprocessing
import requests
import copy


def genenerate_new_individual(rfc):
    pred = []
    for k in rfc.proba(ET):
        pred.append(np.random.choice(list(range(len(k))), 1, p=k)[0])
    return pred

###################SVM######################
jobs = 512 
resources = 16

u = Utils()
ET, CT, maquinas = u.initialize('512x16/u_c_lolo.0', jobs, resources)
m, i = u.minmin(ET, CT, maquinas)

start_time = time.time()
rfc = RandomForestClass(ET, i)
rfc.create()
rfc.fit()
#print(rfc.pred(ET))

preds = [genenerate_new_individual(rfc) for i in range(1000)]
individuals = []

for p in preds:
    temp = Individual(ET,p)
    for k in range(1):
        temp_cp = copy.deepcopy(temp)
        temp_cp.local_search_for_vs()
        if temp_cp.fitness < temp.fitness:
            temp = temp_cp
    individuals.append(temp)

individuals_s = sorted(individuals, key=lambda x:x.fitness)
print("--- %s seconds ---" % (time.time() - start_time))
print(individuals_s[0].fitness)
print(individuals_s[-1].fitness)
individual = Individual(ET,i)
print(individual.fitness)