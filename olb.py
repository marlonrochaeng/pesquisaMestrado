import numpy as np
from utils import Utils

jobs = 512 
resources = 16

u = Utils()

ET, CT, maquinas = u.initialize('512x16/u_c_lohi.0', jobs, resources)


res, individuo = u.olb(ET, CT, maquinas)
res.sort()
print("Maquinas: ",res)