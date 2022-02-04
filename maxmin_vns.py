from tkinter import E
import numpy as np
from utils import Utils
import random
import copy
from joblib import Parallel, delayed
import time
from multiprocessing import Process, Queue


def get_max_in_array(array):
        '''
        Esta função a posição do menor elemento do array passado por parametro
        '''
        return np.where(array == array.max())[0][0]

def get_fitness(ET, individual):
        maquinas = np.zeros(ET.shape[1])
        
        for i in range(ET.shape[0]):
            maquinas[individual[i]] += ET[i][individual[i]]

        return maquinas[get_max_in_array(maquinas)]

def get_fitness_and_machines(ET, individual):
        maquinas = np.zeros(ET.shape[1])
        
        for i in range(ET.shape[0]):
            maquinas[individual[i]] += ET[i][individual[i]]

        return maquinas[get_max_in_array(maquinas)], maquinas

def stochastic_1_opt(ET, city_tour):
    best_route = copy.deepcopy(city_tour)      
    i, j  = random.sample(range(0, len(city_tour)-1), 2)          
    best_route[i] = best_route[j]           
    makespan = get_fitness(ET, best_route)                   
    return best_route, makespan



def stochastic_3_opt(ET, city_tour):
    best_route = copy.deepcopy(city_tour)               
    makespan, machines = get_fitness_and_machines(ET, best_route)

    return local_search_for_vs(machines, best_route, ET)

def get_n_least_loaded_machines(maquinas):
    t_machines = enumerate(maquinas)
    t_machines_ordered = sorted(t_machines, key=lambda x:x[1])
    n_least_loaded_machines = [t_machines_ordered[i][0] for i in range(int(len(t_machines_ordered)/2))]
    return n_least_loaded_machines

def local_search_for_vs(maquinas, individual, ET):
    most_loaded_machine = get_max_in_array(maquinas)
    n_least_loaded_machines = get_n_least_loaded_machines(maquinas)

    random_task = random.choice(np.where(individual == most_loaded_machine)[0])
    random_machine = random.choice(n_least_loaded_machines)

    maquinas[most_loaded_machine] -= ET[random_task][most_loaded_machine]
    maquinas[random_machine] += ET[random_task][random_machine]

    individual[random_task] = random_machine

    #print(f"most_loaded_machine: {most_loaded_machine}, random_task: {random_task}, random_machine: {random_machine}")
    #print(get_fitness(ET, individual))
    return individual, get_fitness(ET, individual)
def stochastic_2_opt(ET, city_tour):
    best_route = copy.deepcopy(city_tour)      
    i, j  = random.sample(range(0, len(city_tour)-1), 2)          
    best_route[i], best_route[j]  = best_route[j], best_route[i]           
    makespan = get_fitness(ET, best_route)                   
    return best_route, makespan
    
# Function: Local Search
def local_search(ET, city_tour, max_attempts = 50, neighbourhood_size = 5):
    count = 0
    solution = copy.deepcopy(city_tour)
    sol_makespan = get_fitness(ET,solution)
    while (count < max_attempts): 
        for i in range(0, neighbourhood_size):
            candidate, cand_makespan = stochastic_2_opt(ET, city_tour = solution)
        if cand_makespan < sol_makespan:
            solution  = copy.deepcopy(candidate)
            count = 0
        else:
            count = count + 1                             
    return solution 

# Function: Variable Neighborhood Search
def variable_neighborhood_search(ET, city_tour, max_attempts = 5, neighbourhood_size = 5, iterations = 50):
    count = 0
    solution = copy.deepcopy(city_tour)
    best_solution = copy.deepcopy(city_tour)
    #print("ET:", ET)
    #print("Best Sol:", best_solution)
    best_sol_makespan = get_fitness(ET, best_solution)
    while (count < iterations):
        for i in range(0, neighbourhood_size):
            for j in range(0, neighbourhood_size):
                #solution, _ = stochastic_2_opt(ET, city_tour = best_solution)
                solution = local_search(ET, city_tour = solution, max_attempts = max_attempts, neighbourhood_size = neighbourhood_size )
                sol_makespan = get_fitness(ET,solution)
                if (sol_makespan < best_sol_makespan):
                    best_solution = copy.deepcopy(solution) 
                    best_sol_makespan = get_fitness(ET,best_solution)
                    #print("-------------------")
                    #print("solution makespan:", sol_makespan)
                    #print("best solution makespan:", best_sol_makespan)
                    break
        count = count + 1
        #print("Iteration = ", count)
        #print(get_fitness(ET, best_solution))
    return best_solution, best_sol_makespan

def getNeighbours(solution):
    neighbours = []
    for i in range(int(len(solution)/5)):
        for j in range(i + 1, int(len(solution)/5)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)
    print(f"len neighbours: {len(neighbours)}")
    return neighbours

def get_neighbour(solution, i, j):
    neighbour = solution.copy()
    neighbour[i] = solution[j]
    neighbour[j] = solution[i]
    return neighbour


def getNeighboursParallel(solution):
    neighbours = Parallel(n_jobs=4)(delayed(get_neighbour)(solution, i,j) for i in range(int(len(solution)/5)) for j in range(int(len(solution)/5)))
    return neighbours

def getBestNeighbour(ET, neighbours):
    bestRouteLength = get_fitness(ET, neighbours[0])
    bestNeighbour = neighbours[0]
    for neighbour in neighbours:
        currentRouteLength = get_fitness(ET, neighbour)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
    return bestNeighbour, bestRouteLength

def hillClimbing(ET, individual):
    currentSolution = individual
    currentRouteLength = get_fitness(ET,individual)
    neighbours = getNeighbours(currentSolution)
    bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(ET, neighbours)

    while bestNeighbourRouteLength < currentRouteLength:
        currentSolution = bestNeighbour
        currentRouteLength = bestNeighbourRouteLength
        neighbours = getNeighbours(currentSolution)
        bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(ET, neighbours)

    return currentSolution, currentRouteLength
    

jobs = 512 
resources = 16

u = Utils()

ET, CT, maquinas = u.initialize('512x16/u_c_lolo.0', jobs, resources)

res, individuo = u.maxmin2(ET, CT, maquinas)
res.sort()
print("Maquinas: ",res[-1])
#print("Individuo: ",individuos)
#print(variable_neighborhood_search(ET, individuo))
start = time.time()
print(hillClimbing(ET, individuo))
print("Maquinas: ",res[-1])
end = time.time()
print(f"elapsed time: {end - start}")