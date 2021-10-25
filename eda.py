from typing import Any
import numpy as np
from individual import Individual
import random
import pandas as pd
import os.path as path
import time
import numpy.random as npr
from joblib import Parallel, delayed
from utils import Utils
import json
import copy


class EDA():
    def __init__(self, jobs, machines, numInd, numGen, ET, toMatrix, elitism, path, mutation) -> None:
        self.jobs = jobs
        self.machines = machines
        self.numInd = numInd
        self.numGen = numGen
        self.prob_matrix = self.create_prob_matrix()
        self.ET = ET
        self.gen = []
        self.best_makespan = None
        self.to_matrix = toMatrix
        self.elitism = elitism
        self.selection_method = 'roulette wheel'
        self.path = path
        self.mutation = mutation

    def select_with_choice(self, population):
        max = sum([c for c in population])
        selection_probs = [c/max for c in population]
        return npr.choice(len(population), p=selection_probs)
    
    def tournament_selection(self, lista):
        k = int(len(lista)/3)

        tournament_list = []
        enum_list = list(enumerate(lista))

        for _ in range(k):
            tournament_list.append(random.choice(enum_list))
        
        tournament_list = sorted(tournament_list, key=lambda x:x[1], reverse=True)
        #print(tournament_list)
        return tournament_list[0][0]
    

    def linear_ranking(self, i):
        #index, maquina, novo_index_ordenado, probabilidade
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
  
    def create_prob_matrix(self) -> Any:
        return np.zeros((self.jobs, self.machines))

    def fill_prob_matrix(self, individuals):
        for i in individuals:
            for j in range(len(i.individual)):
                self.prob_matrix[j][i.individual[j]] += 1
    
    def fill_single_prob_matrix(self, individual):
        for j in range(len(individual.individual)):
            self.prob_matrix[j][individual.individual[j]] += 1
    
    def paralel_gen(self, qtd_individuals):
        new_gen =Individual(self.ET, self.create_new_individual())
        return new_gen
    
    def create_new_individual(self):
        new_individual = np.random.randint(0, 0, 0)
        count = 0

        for i in self.prob_matrix:
            count += 1
            if self.selection_method == 'roulette wheel':
                choice = self.select_with_choice(i)
            elif self.selection_method == 'tournament':
                choice = self.tournament_selection(i)#, int(self.machines/3))
                #print("I:",i)
                #print("choice:",choice)
                #input()
            elif self.selection_method == 'linear ranking':
                choice = self.linear_ranking(i)        

            new_individual = np.append(new_individual, choice)

        return new_individual
    
    def create_first_gen(self):
        self.gen = []
        
        u = Utils()
        ET, CT, maquinas = u.initialize('512x16/'+self.path, self.jobs, self.machines)

        res, individuos = u.maxmin2(ET, CT, maquinas)
        self.gen.append(Individual(self.ET.copy(), individuos, 'maxmin'))

        res, individuos = u.minmin(ET, CT, maquinas)
        self.gen.append(Individual(self.ET.copy(), individuos, 'minmin'))

        res, individuos = u.mct2(ET, CT, maquinas)
        self.gen.append(Individual(self.ET.copy(), individuos, 'mct'))

        res, individuos = u.met(ET, CT, maquinas)
        self.gen.append(Individual(self.ET.copy(), individuos, 'met'))

        res, individuos = u.olb(ET, CT, maquinas)
        self.gen.append(Individual(self.ET.copy(), individuos, 'olb'))
        
        #for _ in range(self.numInd):
        #    self.gen.append(Individual(self.ET.copy()))

        #self.save_to_json(100)

        pop = json.load(open('population_100.json'))

        #população gerada por força bruta
        
        _path = self.path.split('.')[0]
        u_c_pop = json.load(open(f'population_map-{_path}.txt.json'))
        print(f"--------population_map-{_path}.txt.json---------")

        #for i in range(len(u_c_pop)):
        #    self.gen.append(Individual(self.ET.copy(), u_c_pop[str(i)]))
        
        #população gerada atraves da populacao controle
        for i in range(len(pop)):
            self.gen.append(Individual(self.ET.copy(), pop[str(i)]))
        
        self.first_gen_len = len(self.gen)
    
    def save_to_json(self, num):
        
        pop = {}
        count = 0

        for i in self.gen:
            pop[count] = []
            for j in i.individual:
                pop[count].append(int(j))
            count += 1
        
        with open(f"population_{num}.json", "w") as outfile: 
            json.dump(pop, outfile)
        
    
    def order_pop(self, arr):
        return sorted(arr, key=lambda x: x.fitness)

    def form_new_gen(self, to_matrix_percent):
        self.mutate()
        #self.vs_local_search()
        
        self.gen = self.order_pop(self.gen)

        qtd_individuals = int(self.elitism*len(self.gen)/100)
        new_gen = self.gen[:qtd_individuals].copy()

        self.gen = self.gen[:int(len(self.gen)*to_matrix_percent)]
        self.prob_matrix = self.create_prob_matrix()
        #self.fill_prob_matrix(self.gen)
        self.v_func = np.vectorize(self.fill_single_prob_matrix)
        self.v_func(self.gen)

        new_gen += Parallel(n_jobs=8)(delayed(self.paralel_gen)(i) for i in range(self.numInd - qtd_individuals))
        print("New ind:",self.numInd - qtd_individuals)
        print("Num ind:",self.numInd)
        self.gen = new_gen.copy()
        print("Num ind:",len(self.gen))

        temp = self.order_pop(new_gen)
        print("---------------------")
        print("Heuristica que permaneceu nessa geração:")
        for i in temp:
            if i.heuristic is not None:
                print(i.heuristic)
        print("---------------------")
        print("Worst individul makespan:")
        print(temp[-1].fitness)
        if self.best_makespan is None:
            self.best_makespan = temp[-1].fitness
        else:
            self.best_makespan = temp[-1].fitness if temp[-1].fitness < self.best_makespan else self.best_makespan

    def save_to_csv(self):

        if path.exists('resultados/EDA_SEM_POP_GRANDE_E_COM_HEURISITCA.csv'):
            df_results = pd.read_csv('resultados/EDA_SEM_POP_GRANDE_E_COM_HEURISITCA.csv', header=0, index_col=0)
        else:
            columns = ['jobs','machines','numInd','numGen','makespan', 'to_matrix_percentage']
            df_results = pd.DataFrame(columns=columns)

        df_results = df_results.append(
            {'jobs': self.jobs,
             'machines': self.machines,
             'numInd': self.numInd,
             'numGen': self.numGen,
             'makespan': self.best_makespan,
             'to_matrix_percentage': self.to_matrix,
             'exec_time': self.exec_time,
             'selection_method': self.selection_method,
             'elitismo': self.elitism,
             'instance': self.path,
             'mutation':self.mutation}, 
                        ignore_index=True)   
        df_results.to_csv('resultados/EDA_SEM_POP_GRANDE_E_COM_HEURISITCA.csv')     
        df_results = df_results.loc[:, ~df_results.columns.str.contains('^Unnamed')]
    
    def vs_local_search(self):
        for g in self.gen:
            g_cp = copy.deepcopy(g)
            g_cp.local_search_for_vs()
            if g_cp.fitness < g.fitness:
                g = g_cp

    def mutate(self):
        for i in self.gen:
            p = random.randint(1,100)
            if p <= self.mutation and i.heuristic is None:
                pos = random.randint(0,len(self.gen[0].individual)-1)
                i.individual[pos] = random.randint(0, self.machines - 1)
                i.fitness = i.get_fitness() 

    def mutate_swap(self):
        for i in self.gen:
            p = random.randint(1,100)
            if p <= self.mutation and i.heuristic is None:
                pos1 = random.randint(0,len(self.gen[0].individual)-1)
                pos2 = random.randint(0,len(self.gen[0].individual)-1)

                while pos1 == pos2 or i.individual[pos1] == i.individual[pos2] :
                    pos2 = random.randint(0,len(self.gen[0].individual)-1)

                i.individual[pos1], i.individual[pos2] = i.individual[pos2], i.individual[pos1]
                i.fitness = i.get_fitness()
    
    def eda_generations(self):
        start_time = time.time()
        self.create_first_gen()

        for i in range(self.numGen):
            print("GEN:",i)
            self.form_new_gen(self.to_matrix)
        print("--- %s seconds ---" % (time.time() - start_time))
        self.exec_time = (time.time() - start_time)
        
        self.save_to_csv()
        
        
  