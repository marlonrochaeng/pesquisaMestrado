import random
import numpy as np
from job import Job
from individual import Individual
from eda import EDA

class Orquestrador():
    def __init__(self, jobs, machines, path, numInd, numGen, toMatrix) -> None:
        self.jobs = jobs
        self.machines = machines
        self.path = path
        self.numInd = numInd
        self.numGen = numGen
        self.ET, self.CT, self.maquinas = self.initialize()
        self.to_matrix = toMatrix
        self.eda = EDA(jobs, machines, numInd, numGen, self.ET, self.to_matrix)
        

    def initialize(self) -> tuple:
        array = np.array(open(self.path).readlines(),dtype=float)
        ET = np.reshape(array,(self.jobs, self.machines))
        CT = ET.copy()
        maquinas = np.zeros(self.machines, dtype=float)
        return ET, CT, maquinas
    
    def run_eda(self):
        self.eda.eda_generations()

