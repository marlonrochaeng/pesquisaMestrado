from svm_info import SVM_info, LibSVM_info
from utils import Utils
from individual import Individual
import numpy as np
import time
import multiprocessing
import requests
from libsvm.commonutil import evaluations

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

svm = LibSVM_info(ET, i)

svc = svm.train()
p_labels, p_acc, p_vals = svm.get_label(i, svc,ET)

#print(p_vals[0])


