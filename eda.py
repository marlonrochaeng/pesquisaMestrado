from typing import Any
import numpy as np
from individual import Individual
import random
import pandas as pd
import os.path as path


class EDA():
    def __init__(self, jobs, machines, numInd, numGen, ET, toMatrix) -> None:
        self.jobs = jobs
        self.machines = machines
        self.numInd = numInd
        self.numGen = numGen
        self.prob_matrix = self.create_prob_matrix()
        self.ET = ET
        self.gen = []
        self.best_makespan = None
        self.to_matrix = toMatrix
    
    def create_prob_matrix(self) -> Any:
        return np.zeros((self.jobs, self.machines))

    def fill_prob_matrix(self, individuals):
        for i in individuals:
            for j in i.individual:
                self.prob_matrix[j][i.individual[j]] += 1
    
    def create_new_individual(self):
        new_individual = np.random.randint(0, 0, 0)
        for i in self.prob_matrix:
            choice = np.where(i == random.choices(i, i, k=1)[0])
            if len(choice) > 1:
                choice = random.choice(choice)

            new_individual = np.append(new_individual, choice)

        return new_individual
    
    def create_first_gen(self):
        self.gen = []
        for _ in range(self.numInd):
            self.gen.append(Individual(self.ET.copy()))

    
    def order_pop(self, arr):
        return sorted(arr, key=lambda x: x.fitness)

    def form_new_gen(self, to_matrix_percent):
        
        self.gen = self.order_pop(self.gen)
        new_gen = []

        self.gen = self.gen[:int(len(self.gen)*to_matrix_percent)]
        self.prob_matrix = self.create_prob_matrix()
        self.fill_prob_matrix(self.gen)

        for _ in range(self.numInd):
            new_gen.append(Individual(self.ET, self.create_new_individual()))
        
        self.gen = new_gen.copy()

        temp = self.order_pop(new_gen)
        print("Worst individul makespan:")
        print(temp[-1].fitness)
        if self.best_makespan is None:
            self.best_makespan = temp[-1].fitness
        else:
            self.best_makespan = temp[-1].fitness if temp[-1].fitness < self.best_makespan else self.best_makespan

    def save_to_csv(self):

        if path.exists('results.csv'):
            df_results = pd.read_csv('results.csv', header=0, index_col=0)
        else:
            columns = ['jobs','machines','numInd','numGen','makespan', 'to_matrix_percentage']
            df_results = pd.DataFrame(columns=columns)

        df_results = df_results.append(
            {'jobs': self.jobs,
             'machines': self.machines,
             'numInd': self.numInd,
             'numGen': self.numGen,
             'makespan': self.best_makespan,
             'to_matrix_percentage': self.to_matrix}, 
                        ignore_index=True)   
        df_results.to_csv('results.csv')     
        df_results = df_results.loc[:, ~df_results.columns.str.contains('^Unnamed')]
        
    
    def eda_generations(self):
        self.create_first_gen()

        for i in range(self.numGen):
            print("GEN:",i)
            self.form_new_gen(0.5)
        
        self.save_to_csv()
        
        


        

    
            