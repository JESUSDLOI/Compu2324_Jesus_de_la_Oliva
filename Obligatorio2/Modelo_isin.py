import numpy as np 
from numpy import random
from numba import jit
import time

# ================================================================================

#ININICIAR VARIABLES

#Lado de la malla
lado_malla = np.full((20, 64)).astype(np.int8)

#Temperatura
temperaturas = np.linspace((0.5, 5, 20)).astype(np.float32)

#Número de pasos_monte
pasos_monte = np.full((20, 100000)).astype(np.int32)

# ================================================================================



#Matriz aleatoria entre estado 1 y -1
@jit(nopython=True, fastmath=True)
def mtrz_aleatoria(M):
    matriz = 2 * np.random.randint(0, 2, size=(M, M)).astype(np.int8) - 1
    return matriz

#Condiciones de contorno periódicas
@jit(nopython=True, fastmath=True)
def cond_contorno_1(M, i, j):
    if i == 0:
        izquierda = M - 2
    else:
        izquierda = i - 1
    if i == M - 1:
        derecha = 1
    else:
        derecha = i + 1
    if j == 0:
        arriba = M - 2
    else:
        arriba = j - 1
    if j == M - 1:
        abajo = 1
    else:
        abajo = j + 1
    return izquierda, derecha, arriba, abajo
    


#Cálculo de la matriz
@jit(nopython=True, fastmath=True)
def calculo_matriz(matriz, M):
    #Iteración sobre la matriz
    i = np.random.randint(0, M)
    j = np.random.randint(0, M)

    #Condiciones de contorno
    izquierda, derecha, arriba, abajo = cond_contorno_1(M, i, j)

    #Calculo de la variación de la energía
    delta_E = 2*matriz[i,j]*(matriz[(derecha),j] + matriz[i,(abajo)] + matriz[(izquierda),j] + matriz[i,(arriba)])
    return i, j, delta_E

#Secuencia de Ising
@jit(nopython=True, fastmath=True)
def secuencia_isin(M, T, matriz):
    
    i, j, delta_E = calculo_matriz(matriz, M)

    #Probabilidad de cambio
    p_0 = np.exp(-2*delta_E/T)

    #Evaluar probabilidad de cambio
    if 1 < p_0:
        p = 1
    else:
        p = p_0

    #Número aleatorio entre  para comparar con la probabilidad
    r = np.random.uniform(0, 1)

    #Comparar probabilidad para cambiar el spin
    if r < p:
        matriz[i,j] = -matriz[i,j]  

    #Guardar matriz en archivo

    return matriz

#Matriz de Ising
def ising_model(M, T, N):
    #Matriz de Ising
    matriz = mtrz_aleatoria(M)
        #Archivo de datos
    with open('ising_data_tem_{0}_malla_{1}.dat'.format(T, M), 'w') as file:
        for n in range(N):
            for k in range(M**2):
                #Matriz resultado
                matriz = secuencia_isin(M, T, matriz)
                #Guardar matriz en archivo
            if n % 10 == 0:
                file.write('\n')
                np.savetxt(file, matriz, fmt='%d', delimiter=',') 


#Simulaciones de Monte Carlo distintas temperaturas y mallas
def simulaciones(lado_malla, temperaturas, pasos_monte):
    #Cantidad de archivos
    C = len(temperaturas)
    resultados = np.zeros((C, 3))

    i = 0
    while i < C:
        #Temperatura y lado de la malla
        T = temperaturas[i]
        M = lado_malla[i]
        N = pasos_monte[i]
        #Modelo y tiempo de ejecución
        tiempo_0 = time.time()
        ising_model(M, T, N)
        tiempo_1 = time.time()
        tiempo = tiempo_1 - tiempo_0

        #Guardar parámetros de la simulación
        resultados[i, 0] = T
        resultados[i, 1] = M
        resultados[i, 2] = tiempo
        i += 1

    return resultados


print(simulaciones(lado_malla, temperaturas, pasos_monte))



