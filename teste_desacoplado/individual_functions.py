import random
import numpy as np
from numba import jit



def create_individual(ET, individual=None, heuristic=None) -> None:
    if individual is None:
        individual = generate_random_individual(ET)
    fitness = get_fitness(ET, individual)
    heuristic = heuristic
    return individual, fitness, heuristic

def get_fitness(ET, individual):
    maquinas = np.zeros(ET.shape[1])

    for i in range(ET.shape[0]):     
        maquinas[individual[i]] += ET[i][individual[i]]
    
    return maquinas[get_max_in_array(maquinas)]

def generate_random_individual(ET):
    return np.random.randint(0, ET.shape[1], ET.shape[0])

def get_max_in_array(array):
    '''
    Esta função a posição do menor elemento do array passado por parametro
    '''
    return np.where(array == array.max())[0][0]
    
    
def get_min_in_array(maquinas):
    '''
    Esta função a posição do menor elemento do array passado por parametro
    '''
    return np.where(maquinas == maquinas.min())[0][0]
    
def get_max_el_in_array(self):
    '''
    Esta função a posição do menor elemento do array passado por parametro
    '''
    return np.where(self.maquinas == self.maquinas.max())[0][0]
    

       


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