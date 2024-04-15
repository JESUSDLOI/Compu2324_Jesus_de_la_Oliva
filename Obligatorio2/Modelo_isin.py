import numpy as np 
from numpy import random
import scipy as sp
from numba import jit

#Matriz aleatoria entre estado 1 y -1
def mtrz_aleatoria(N):
    matriz = np.random.choice([-1,1],(N,N))
    return matriz

#Matriz de Ising
def ising_model(N, T):
    #Matriz de Ising
    matriz = mtrz_aleatoria(N)

    #Condiciones de contorno periódicas
    

    for k in range(N**2):
        #Iteración sobre la matriz
        i = np.random.randint(0,N)
        j = np.random.randint(0,N)

        #Calculo de la variación de la energía
        delta_E = 2*matriz[i,j]*(matriz[(i+1),j] + matriz[i,(j+1)] + matriz[(i-1),j] + matriz[i,(j-1)])
        
        #Evaluar probabilidad de cambio
        p = np.min([1, np.exp(-2*delta_E/T)])

        #Número aleatorio entre  para comparar con la probabilidad
        r = np.random.rand(0,1)

        if r < p:
            matriz[i,j] = -matriz[i,j]  

    return matriz


T=5
while T > 0:
    T = T - 1
    print(ising_model(5, T))