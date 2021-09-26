import numpy as np
import json

def big_pop_to_json(text_file):
    pop = open(text_file,'r').readlines()
    _pop = {}
    for i in range(len(pop)):
        temp = pop[i].replace('  ',' ')
        temp = temp.split(' ')
        temp[-1] = temp[-1].replace('\n','')

        temp = list(filter(lambda x : x != ' ' and x != '', temp))
        for j in range(len(temp)):
            
            temp[j] = int(temp[j]) - 1
    
        _pop[i] = temp

    with open(f"population_{text_file}.json", "w") as outfile: 
            json.dump(_pop, outfile)

        
maps = [
'map-u_c_hihi.txt',
'map-u_c_hilo.txt',
'map-u_c_lohi.txt',
'map-u_c_lolo.txt',
'map-u_i_hihi.txt',
'map-u_i_hilo.txt',
'map-u_i_lohi.txt',
'map-u_i_lolo.txt',
'map-u_s_hihi.txt',
'map-u_s_hilo.txt',
'map-u_s_lohi.txt',
'map-u_s_lolo.txt'
]

for m in maps:
    big_pop_to_json(m)