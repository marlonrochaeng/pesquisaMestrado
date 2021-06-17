import random
import numpy as np
from multiprocessing import Pool
import time




def select_with_choice(population):
        max = sum([c for c in population])
        selection_probs = [c/max for c in population]
        return np.random.choice(len(population), p=selection_probs)
    
def tournament_selection(lista):
    k = int(len(lista)/3)

    tournament_list = []
    enum_list = list(enumerate(lista))

    for _ in range(k):
        tournament_list.append(random.choice(enum_list))
    
    tournament_list = sorted(tournament_list, key=lambda x:x[1], reverse=True)
    return tournament_list[0][0] , tournament_list[0][1] 

def linear_ranking(i):
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
        
        return np.random.choice([v[0] for v in ordered_values], p=p)



if __name__ == '__main__':
    vetor = np.random.randint(1,1000,100000)
    maiores = sorted(vetor, reverse=True)[0:3]

    print("Vetor gerado:",vetor)
    print("3 maiores elementos:", maiores)
    count_maior = 0
    count_menor = 0
    outputs = []
    start_time = time.time()
    for i in range(1000):
        outputs.append(select_with_choice(vetor))
    print("Out:", outputs)
    print("--- %s seconds ---" % (time.time() - start_time))

    for i in range(100):
        with Pool(8) as p:
            pos = p.map(select_with_choice,vetor)
            if vetor[pos] in maiores:
                count_maior += 1
            else:
                count_menor += 1
    
    print("Quantidade de vezes que um dos 3 melhores foi selecionado:",count_maior)
    print("Quantidade de vezes que os 3 melhores nao foram selecionados: ",count_menor)