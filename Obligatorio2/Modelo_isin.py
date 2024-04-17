import numpy as np 
from numpy import random
from numba import jit
import time

#Matriz aleatoria entre estado 1 y -1
@jit(nopython=True)
def mtrz_aleatoria(N):
    matriz = np.random.choice(2,(N,N))
    matriz = np.where(matriz==0, -1, matriz)
    return matriz

#Condiciones de contorno periódicas
jit(nopython=True)
def cond_contorno(matriz, N):
    matriz = np.pad(matriz, pad_width=2, mode='wrap')
    matriz = np.delete(matriz, [1, N+1], axis=0)
    matriz = np.delete(matriz, [1, N+1], axis=1)
    return matriz

#Cálculo de la matriz
jit(nopython=True)
def calculo_matriz(matriz, N, T):
    #Iteración sobre la matriz
    i = np.random.randint(1,N+1)
    j = np.random.randint(1,N+1)

    #Calculo de la variación de la energía
    delta_E = 2*matriz[i,j]*(matriz[(i+1),j] + matriz[i,(j+1)] + matriz[(i-1),j] + matriz[i,(j-1)])
    return i, j, delta_E

#Secuencia de Ising
jit(nopython=True)
def secuencia_isin(N, T, matriz):
    
    i, j, delta_E = calculo_matriz(matriz, N, T)

    #Evaluar probabilidad de cambio
    p = np.min([1, np.exp(-2*delta_E/T)])

    #Número aleatorio entre  para comparar con la probabilidad
    r = np.random.uniform(0, 1)

    #Comparar probabilidad para cambiar el spin
    if r < p:
        matriz[i,j] = -matriz[i,j]  

    #Guardar matriz en archivo
    submatriz = matriz[1:N+1, 1:N+1]  
    
    return submatriz   

#Matriz de Ising
def ising_model(N, T):

    #Matriz de Ising
    matriz = mtrz_aleatoria(N)

    #Condiciones de contorno
    matriz = cond_contorno(matriz, N)

    #Archivo de datos
    with open('ising_data.dat', 'w') as file:

        for k in range(N*10000):
            #Matriz resultado
            submatriz = secuencia_isin(N, T, matriz)
            #Guardar matriz en archivo
            file.write('\n')
            np.savetxt(file, submatriz, fmt='%d', delimiter=',') 
    file.close()

tiempo_0 = time.time()

ising_model(10, 0.4)

tiempo_1 = time.time()

print('Tiempo de ejecución: ', tiempo_1 - tiempo_0)

