import argparse, sys
import os
import time
import numpy as np
from functions import *

parser=argparse.ArgumentParser()

parser.add_argument('--jobs', help='number of jobs')
parser.add_argument('--machines', help='number of jobs to schedule')
parser.add_argument('--path', help='path for the jobs instance')
parser.add_argument('--numInd', help='number of individuals')
parser.add_argument('--numGen', help='number of generations')
parser.add_argument('--toMatrix', help='percentual da matriz')
parser.add_argument('--elitism', help='percentual de individuos que passam de Geração')
parser.add_argument('--mutation', help='percentual de individuos que sofrem mutacao')



args=parser.parse_args()

jobs = [int(i) for i in args.jobs.split(',')]
machines = [int(i) for i in args.machines.split(',')]
numInd = [int(i) for i in args.numInd.split(',')]
numGen = [int(i) for i in args.numGen.split(',')]
toMatrix = [float(i) for i in args.toMatrix.split(',')]
path = [i for i in args.path.split(',')]
elitism = [int(i) for i in args.elitism.split(',')]
mutation = [int(i) for i in args.mutation.split(',')]


for j in jobs:
    for m in machines:
        for ni in numInd:
            for ng in numGen:
                for tm in toMatrix:
                    for p in path:
                        for e in elitism:

                            best_makespan = None
                            for mu in mutation:

                                array = np.array(open(p).readlines(),dtype=float)
                                ET = np.reshape(array,(j, m))
                                CT = ET.copy()
                                maquinas = np.zeros(m, dtype=float)

                                start_time = time.time()
                                first_gen = create_first_gen(p,j,m,ET,CT)
                                for i in range(ng):
                                    print("GEN:",i)
                                    form_new_gen(first_gen, e, tm, best_makespan, ni, ET, mu, m, j)
