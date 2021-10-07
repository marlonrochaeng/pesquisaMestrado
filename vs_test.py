from svm_info import SVM_info
from utils import Utils
from individual import Individual
import numpy as np
import time
import multiprocessing
import requests

def generate_prediction(svm):
    pred = []
    for k in svm.clf.predict_proba(ET):
        pred.append(np.random.choice(list(range(len(k))), 1, p=k)[0])
    return pred


###################SVM######################
jobs = 512 
resources = 16

u = Utils()
ET, CT, maquinas = u.initialize('512x16/u_c_lolo.0', jobs, resources)
m, i = u.minmin(ET, CT, maquinas)

svm = SVM_info(ET, i)
print(i)
svm.create_classifier()
svc = svm.train()
###################SVM######################
start_time = time.time()
for s in svm.clf.predict_proba(ET):
    print(s)
    input()
individuals = []
preds = [generate_prediction(svm) for i in range(1000)]
for p in preds:
    temp = Individual(ET,p)
    print(temp.fitness)
    temp.local_search_for_vs()
    print(temp.fitness)
    input()
    individuals.append(temp)

individuals_s = sorted(individuals, key=lambda x:x.fitness)
print("--- %s seconds ---" % (time.time() - start_time))
print(individuals_s[0].fitness)

individual = Individual(ET,i)
print(individual.fitness)
#print(individual.maquinas)

