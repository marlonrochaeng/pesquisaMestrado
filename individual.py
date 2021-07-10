import random
import numpy as np


class Individual():
    def __init__(self, ET, individual=None, heuristic=None) -> None:
        self.individual = individual 
        if self.individual is None:
            self.individual = self.generate_random_individual(ET)
        self.ET = ET
        self.fitness = self.get_fitness()
        self.heuristic = heuristic

    def generate_random_individual(self, ET):
        return np.random.randint(0, ET.shape[1], ET.shape[0])

    @staticmethod
    def get_max_in_array(array):
        '''
        Esta função a posição do menor elemento do array passado por parametro
        '''
        return np.where(array == array.max())[0][0]
    
    def get_fitness(self):
        maquinas = np.zeros(self.ET.shape[1])
        
        #print("individual:", self.individual)

        for i in range(self.ET.shape[0]):
            
            maquinas[self.individual[i]] += self.ET[i][self.individual[i]]
        
        return maquinas[self.get_max_in_array(maquinas)]

        

