from numpy.core.fromnumeric import size
from eda import EDA
import time
import pandas as pd
import os.path as path
from joblib import Parallel, delayed
import numpy as np
import random

class GA(EDA):
    def __init__(self, jobs, machines, numInd, numGen, ET, toMatrix, elitism, path, mutation) -> None:
        super().__init__(jobs, machines, numInd, numGen, ET, toMatrix, elitism, path)
        self.mutation = mutation

    
    def form_new_gen(self):
        
        #self.gen = self.order_pop(self.gen)

        qtd_individuals = int(self.elitism*len(self.gen)/100)
        new_gen = []

        new_gen += Parallel(n_jobs=4)(delayed(self.paralel_gen)(i) for i in range(self.first_gen_len))

        self.gen += new_gen

        for i in self.gen:
            p = random.randint(1,100)
            if p <= self.mutation and i.heuristic is None:
                pos = random.randint(0,len(self.gen[0].individual)-1)
                i.individual[pos] = random.randint(0, self.machines - 1)
                i.fitness = i.get_fitness()

        self.gen = self.order_pop(self.gen)[:self.first_gen_len]

        print("Heuristica que permaneceu nessa geração:")
        for i in self.gen:
            if i.heuristic is not None:
                print(i.heuristic)
        print("---------------------")
        print("Worst individul makespan:")
        print(self.gen[-1].fitness)

        if self.best_makespan is None:
            self.best_makespan = self.gen[-1].fitness
        else:
            self.best_makespan = self.gen[-1].fitness if self.gen[-1].fitness < self.best_makespan else self.best_makespan


    def roulette_wheel_selection(self):
        population_fitness = sum([chromosome.fitness for chromosome in self.gen])
        chromosome_probabilities = [chromosome.fitness/population_fitness for chromosome in self.gen]        
        chromosome_probabilities = (1 - np.array(chromosome_probabilities)) / (len(self.gen) - 1)
        
        result = np.random.choice(self.gen, size=2, p=chromosome_probabilities)

        return result

    def single_point_crossover(self, p1, p2, x=None):
        if x is None:
            x = random.randint(1,len(self.gen[0].individual)-2)
        filho1 = np.append(p1.individual[:x].copy(),p2.individual[x:].copy())
        filho2 = np.append(p2.individual[:x].copy(),p1.individual[x:].copy())
        
        p = random.randint(1,100)

        if p <= 50:
            return filho1
        return filho2
    
    def two_point_crossover(self, p1, p2):

        p1 = list(p1.individual)
        p2 = list(p2.individual)
        
        x1 = random.randint(1,len(self.gen[0].individual)-2)
        x2 = random.randint(1,len(self.gen[0].individual)-2)
        while x1 == x2:
            x2 = random.randint(1,len(self.gen[0].individual)-2)
        
        if x2 > x1:
            x1, x2 = x2, x1

        return p1[:x1] + p2[x1:x2] + p1[x2:]
        


    def create_new_individual(self):
        p1, p2 = self.roulette_wheel_selection()
        result = self.single_point_crossover(p1,p2)
        return result


    def save_to_csv(self):

        if path.exists('AG.csv'):
            df_results = pd.read_csv('AG.csv', header=0, index_col=0)
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
        df_results.to_csv('AG.csv')     
        df_results = df_results.loc[:, ~df_results.columns.str.contains('^Unnamed')]

    def ag_generations(self):
        start_time = time.time()
        self.create_first_gen()

        for i in range(self.numGen):
            print("GEN:",i)
            self.form_new_gen()
        print("--- %s seconds ---" % (time.time() - start_time))
        self.exec_time = (time.time() - start_time)
        
        self.save_to_csv()

