import random
import numpy as np
from utils import Utils


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
    
    
    def get_min_in_array(self):
        '''
        Esta função a posição do menor elemento do array passado por parametro
        '''
        return np.where(self.maquinas == self.maquinas.min())[0][0]
    
    def get_max_el_in_array(self):
        '''
        Esta função a posição do menor elemento do array passado por parametro
        '''
        return np.where(self.maquinas == self.maquinas.max())[0][0]
    
    def get_fitness(self):
        maquinas = np.zeros(self.ET.shape[1])
        
        #print("individual:", self.individual)

        for i in range(self.ET.shape[0]):
            
            maquinas[self.individual[i]] += self.ET[i][self.individual[i]]
        
        self.maquinas = maquinas
        return maquinas[self.get_max_in_array(maquinas)]
    
    def get_n_least_loaded_machines(self):
        t_machines = enumerate(self.maquinas)
        t_machines_ordered = sorted(t_machines, key=lambda x:x[1])
        n_least_loaded_machines = [t_machines_ordered[i][0] for i in range(int(len(t_machines_ordered)/2))]
        return n_least_loaded_machines

    def local_search_for_vs(self):
        most_loaded_machine = self.get_max_el_in_array()
        n_least_loaded_machines = self.get_n_least_loaded_machines()

        random_task = random.choice(np.where(self.individual == most_loaded_machine)[0])
        random_machine = random.choice(n_least_loaded_machines)

        self.maquinas[most_loaded_machine] -= self.ET[random_task][most_loaded_machine]
        self.maquinas[random_machine] += self.ET[random_task][random_machine]

        self.individual[random_task] = random_machine

        #print(f"most_loaded_machine: {most_loaded_machine}, random_task: {random_task}, random_machine: {random_machine}")

        self.fitness = self.get_fitness()
       


'''
u = Utils()
ET, CT, maquinas = u.initialize('512x16/u_c_lolo.0', 512, 16)

i = Individual(ET)
print(i.fitness)
print(sorted(i.maquinas))
print("----------------------------------------")
i.local_search_for_vs()
print(i.fitness)
print(sorted(i.maquinas))
'''