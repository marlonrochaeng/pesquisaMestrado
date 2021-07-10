import random
import numpy as np
from job import Job
from individual import Individual
from ag import GA

class Orquestrador():
    def __init__(self, jobs, machines, path, numInd, numGen, toMatrix, elitism, mutation) -> None:
        self.jobs = jobs
        self.machines = machines
        self.path = path
        self.numInd = numInd
        self.numGen = numGen
        self.ET, self.CT, self.maquinas = self.initialize()
        self.to_matrix = toMatrix
        self.elitism = elitism
        self.ga = GA(jobs, machines, numInd, numGen, self.ET, self.to_matrix, self.elitism, self.path.split('/')[1], mutation)
        
        

    def initialize(self) -> tuple:
        array = np.array(open(self.path).readlines(),dtype=float)
        ET = np.reshape(array,(self.jobs, self.machines))
        CT = ET.copy()
        maquinas = np.zeros(self.machines, dtype=float)
        return ET, CT, maquinas
    
    def run_GA(self):
        self.ga.ag_generations()

