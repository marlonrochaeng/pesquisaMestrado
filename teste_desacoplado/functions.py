from typing import Any
import numpy as np
import random
import pandas as pd
import os.path as path
import time
import numpy.random as npr
from joblib import Parallel, delayed
from utils import Utils
import json
import copy
from individual_functions import *

def create_prob_matrix(jobs, machines) -> Any:
    return np.zeros((jobs, machines))

def fill_prob_matrix(prob_matrix, individuals):
    for i in individuals:
        for j in range(len(i[0])):
            prob_matrix[j][i[0][j]] += 1
    return prob_matrix

def select_with_choice(population):
    max = sum([c for c in population])
    selection_probs = [c/max for c in population]
    return npr.choice(len(population), p=selection_probs)

def tournament_selection(lista):
    k = int(len(lista)/3)

    tournament_list = []
    enum_list = list(enumerate(lista))

    for _ in range(k):
        tournament_list.append(random.choice(enum_list))
    
    tournament_list = sorted(tournament_list, key=lambda x:x[1], reverse=True)
    return tournament_list[0][0]


def linear_ranking(i):
    _sum = sum([j+1 for j in range(len(i))])

    values = [[k,v] for k,v in enumerate(i)]
    ordered_values = sorted(values, key = lambda t: t[1]) 
    count = 1
    p = []

    for j in ordered_values:
        j += [count]
        p.append(count/_sum)
        count += 1
    
    return npr.choice([v[0] for v in ordered_values], p=p)

def fill_single_prob_matrix(prob_matrix, individual):
    for j in range(len(individual)):
        print("individual: ",len(individual))
        print("individual[j]: ",individual[j])
        print("prob_matrix: ",prob_matrix)
        prob_matrix[j][individual[j]] += 1


def paralel_gen(prob_matrix):
    new_gen =create_new_individual(prob_matrix, 'roulette wheel')
    return new_gen

def create_new_individual(prob_matrix, selection_method):
    #print(prob_matrix.shape, selection_method)
    #input()
    new_individual = np.random.randint(0, 0, 0)
    count = 0

    for i in prob_matrix:
        count += 1
        if selection_method == 'roulette wheel':
            choice = select_with_choice(i)
        elif selection_method == 'tournament':
            choice = tournament_selection(i)#, int(self.machines/3))
        elif selection_method == 'linear ranking':
            choice = linear_ranking(i)        

        new_individual = np.append(new_individual, choice)

    return new_individual

def create_first_gen(path, jobs, machines, ET, CT):
    gen = []
    
    u = Utils()
    ET, CT, maquinas = u.initialize(path, jobs, machines)

    res, individuos = u.maxmin2(ET, CT, maquinas)
    gen.append(create_individual(ET.copy(), individuos, 'maxmin'))

    res, individuos = u.minmin(ET, CT, maquinas)
    gen.append(create_individual(ET.copy(), individuos, 'minmin'))

    res, individuos = u.mct2(ET, CT, maquinas)
    gen.append(create_individual(ET.copy(), individuos, 'mct'))

    res, individuos = u.met(ET, CT, maquinas)
    gen.append(create_individual(ET.copy(), individuos, 'met'))

    res, individuos = u.olb(ET, CT, maquinas)
    gen.append(create_individual(ET.copy(), individuos, 'olb'))
    
    #for _ in range(self.numInd):
    #    self.gen.append(Individual(self.ET.copy()))

    #self.save_to_json(100)

    pop = json.load(open('population_100.json'))

    #população gerada por força bruta
    
    _path = path.split('.')[0]
    u_c_pop = json.load(open(f'population_map-{_path}.txt.json'.replace('512x16/','')))
    print(f"--------population_map-{_path}.txt.json---------")

    #for i in range(len(u_c_pop)):
    #    gen.append(Individual(ET.copy(), u_c_pop[str(i)]))
    
    #população gerada atraves da populacao controle
    for i in range(len(pop)):
        gen.append(create_individual(ET.copy(), pop[str(i)]))

    return gen

def save_to_json(gen, num):
    
    pop = {}
    count = 0

    for i in gen:
        pop[count] = []
        for j in i.individual:
            pop[count].append(int(j))
        count += 1
    
    with open(f"population_{num}.json", "w") as outfile: 
        json.dump(pop, outfile)
    

def order_pop(arr):
    return sorted(arr, key=lambda x: x[1])

def form_new_gen(gen, elitism, to_matrix_percent, best_makespan, numInd, ET, mutation, machines, jobs):
    mutate(gen, mutation, machines, ET)
    
    gen = order_pop(gen)

    qtd_individuals = int(elitism*len(gen)/100)
    new_gen = gen[:qtd_individuals].copy()

    gen = gen[:int(len(gen)*to_matrix_percent)]
    prob_matrix = create_prob_matrix(jobs, machines)
    #print("------------------")
    #print(prob_matrix)
    #print("------------------")
    fill_prob_matrix(prob_matrix, gen)
    #v_func = np.vectorize(fill_single_prob_matrix)
    #v_func(prob_matrix, gen)
    #print("------------------")
    #print(prob_matrix)
    #print("------------------")
    #input()
    #new_gen += Parallel(n_jobs=8)(delayed(paralel_gen)(prob_matrix) for i in range(numInd - qtd_individuals))
    for i in range(numInd - qtd_individuals):
        ind = create_individual(ET, create_new_individual(prob_matrix, 'roulette wheel'))
        new_gen += [ind]
    print("New ind:",numInd - qtd_individuals)
    print("Num ind:",numInd)
    gen = new_gen.copy()
    print("Num ind:",len(gen))

    temp = order_pop(new_gen)
    print("---------------------")
    print("Heuristica que permaneceu nessa geração:")
    for i in temp:
        if i[2] is not None:
            print(i[2])
    print("---------------------")
    print("Worst individul makespan:")
    print(temp[-1][1])
    if best_makespan is None:
        best_makespan = temp[-1][1]
    else:
        best_makespan = temp[-1][1] if temp[-1][1] < best_makespan else best_makespan

def save_to_csv(jobs, machines, numInd, numGen, best_makespan, to_matrix, exec_time, elitism, path, mutation):

    if path.exists('resultados/EDA_SEM_POP_GRANDE_E_COM_HEURISITCA.csv'):
        df_results = pd.read_csv('resultados/EDA_SEM_POP_GRANDE_E_COM_HEURISITCA.csv', header=0, index_col=0)
    else:
        columns = ['jobs','machines','numInd','numGen','makespan', 'to_matrix_percentage']
        df_results = pd.DataFrame(columns=columns)

    df_results = df_results.append(
        {'jobs': jobs,
            'machines': machines,
            'numInd': numInd,
            'numGen': numGen,
            'makespan': best_makespan,
            'to_matrix_percentage': to_matrix,
            'exec_time': exec_time,
            'selection_method': 'roulette',
            'elitismo': elitism,
            'instance': path,
            'mutation':mutation}, 
                    ignore_index=True)   
    df_results.to_csv('resultados/EDA_SEM_POP_GRANDE_E_COM_HEURISITCA.csv')     
    df_results = df_results.loc[:, ~df_results.columns.str.contains('^Unnamed')]


def mutate(gen, mutation, machines, ET):
    for i in gen:
        p = random.randint(1,100)
        if p <= mutation and i[2] is None:
            temp_i = list(i)
            pos = random.randint(0,len(gen[0][0])-1)
            temp_i[0][pos] = random.randint(0, machines - 1)
            temp_i[1] = get_fitness(ET, temp_i[0]) 
            i = tuple(temp_i)
            

def eda_generations(to_matrix, numGen):
    start_time = time.time()
    create_first_gen()

    for i in range(numGen):
        print("GEN:",i)
        form_new_gen(to_matrix)
    print("--- %s seconds ---" % (time.time() - start_time))
    exec_time = (time.time() - start_time)
    
    save_to_csv()
        
    return exec_time
  