import numpy as np 
from numpy import random
from numba import jit
import time

# ================================================================================

#ININICIAR VARIABLES

#Lado de la malla
lado_malla = (10, 10, 10)

#Temperatura
temperaturas = (1, 2, 3)

#Número de iteraciones
N = 10000

# ================================================================================



#Matriz aleatoria entre estado 1 y -1
@jit(nopython=True)
def mtrz_aleatoria(M):
    matriz = np.random.choice(2,(M,M))
    matriz = np.where(matriz==0, -1, matriz)
    return matriz

#Condiciones de contorno periódicas
jit(nopython=True)
def cond_contorno(matriz, M):
    matriz = np.pad(matriz, pad_width=2, mode='wrap')
    matriz = np.delete(matriz, [1, M+1], axis=0)
    matriz = np.delete(matriz, [1, M+1], axis=1)
    return matriz

#Cálculo de la matriz
jit(nopython=True)
def calculo_matriz(matriz, M, T):
    #Iteración sobre la matriz
    i = np.random.randint(1,M+1)
    j = np.random.randint(1,M+1)

    #Calculo de la variación de la energía
    delta_E = 2*matriz[i,j]*(matriz[(i+1),j] + matriz[i,(j+1)] + matriz[(i-1),j] + matriz[i,(j-1)])
    return i, j, delta_E

#Secuencia de Ising
jit(nopython=True)
def secuencia_isin(M, T, matriz):
    
    i, j, delta_E = calculo_matriz(matriz, M, T)

    #Evaluar probabilidad de cambio
    p = np.min([1, np.exp(-2*delta_E/T)])

    #Número aleatorio entre  para comparar con la probabilidad
    r = np.random.uniform(0, 1)

    #Comparar probabilidad para cambiar el spin
    if r < p:
        matriz[i,j] = -matriz[i,j]  

    #Guardar matriz en archivo
    submatriz = matriz[1:M+1, 1:M+1]  

    return submatriz   

#Matriz de Ising
def ising_model(M, T, N):

    #Matriz de Ising
    matriz = mtrz_aleatoria(M)

    #Condiciones de contorno
    matriz = cond_contorno(matriz, M)

    #Archivo de datos
    with open('ising_data_tem_{0}_malla_{1}.dat'.format(T, M), 'w') as file:

        for k in range(N):
            #Matriz resultado
            submatriz = secuencia_isin(M, T, matriz)
            #Guardar matriz en archivo
            file.write('\n')
            np.savetxt(file, submatriz, fmt='%d', delimiter=',') 
    file.close()


#Simulaciones de Monte Carlo distintas temperaturas y mallas
def simulaciones(lado_malla, temperaturas, N):
    #Cantidad de archivos
    C = len(temperaturas)
    resultados = np.zeros((C, 3))

    for i in range(C):
        #Temperatura y lado de la malla
        T = temperaturas[i]
        M = lado_malla[i]

        #Modelo y tiempo de ejecución
        tiempo_0 = time.time()
        ising_model(M, T, N)
        tiempo_1 = time.time()

        tiempo = tiempo_1 - tiempo_0

        #Guardar parámetros de la simulación
        resultados[i, 0] = T
        resultados[i, 1] = M
        resultados[i, 2] = tiempo

    return resultados

print(simulaciones(lado_malla, temperaturas, N))