import numpy as np 
from numpy import random
import time

def mtrz_aleatoria(N):
    matriz = np.random.randint(0,N,(N,N))
    return matriz

start_time = time.time()

print(mtrz_aleatoria(100000))

end_time = time.time()
execution_time = end_time - start_time

print(f"El programa se ejecutó en: {execution_time} segundos")