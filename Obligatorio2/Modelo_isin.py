import numpy as np 
from numpy import random
from numba import jit
import threading
import time

# ================================================================================

#ININICIAR VARIABLES

#Lado de la malla
lado_malla = np.full(2, 120).astype(np.int8)

#Temperatura
temperaturas = np.linspace(2.27, 5, 2).astype(np.float32)

#Número de pasos_monte
pasos_monte = np.full(2, 10000).astype(np.int32)

# ================================================================================



#Matriz aleatoria entre estado 1 y -1
@jit(nopython=True, fastmath=True, cache=True)
def mtrz_aleatoria(M):
    matriz = 2 * np.random.randint(0, 2, size=(M, M)).astype(np.int8) - 1
    return matriz

#Condiciones de contorno periódicas
@jit(nopython=True, fastmath=True, cache=True)
def cond_contorno(M, i, j):
    if i == 0:
        izquierda = M - 1
    else:
        izquierda = i - 1
    if i == M - 1:
        derecha = 0
    else:
        derecha = i + 1
    if j == 0:
        arriba = M - 1
    else:
        arriba = j - 1
    if j == M - 1:
        abajo = 0
    else:
        abajo = j + 1
    return izquierda, derecha, arriba, abajo

    


#Cálculo de la matriz
@jit(nopython=True, fastmath=True, cache=True)
def calculo_matriz(matriz, M):
    #Iteración sobre la matriz
    i = np.random.randint(0, M)
    j = np.random.randint(0, M)

    #Condiciones de contorno
    izquierda, derecha, arriba, abajo = cond_contorno(M, i, j)

    #Calculo de la variación de la energía
    delta_E = 2*matriz[i,j]*(matriz[(derecha),j] + matriz[i,(abajo)] + matriz[(izquierda),j] + matriz[i,(arriba)])
    return i, j, delta_E

#Secuencia de Ising
@jit(nopython=True, fastmath=True, cache=True)
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
    with open('ising_data_tem_{0:.2f}_malla_{1}.dat'.format(T, M), 'w') as file:
        for n in range(N):
            for k in range(M**2):
                #Matriz resultado
                matriz = secuencia_isin(M, T, matriz)
                #Guardar matriz en archivo
            if n % 10 == 0:
                file.write('\n')
                np.savetxt(file, matriz, fmt='%d', delimiter=',') 
    pass


#Simulaciones de Monte Carlo distintas temperaturas y mallas
def simulaciones(lado_malla, temperaturas, pasos_monte):

    threads = []
    #Cantidad de archivos
    C = len(lado_malla)
    resultados = np.zeros((C, 2))

    i = 0
    while i in range(C):
        #Temperatura y lado de la malla
        T = temperaturas[i]
        M = lado_malla[i]
        N = pasos_monte[i]
        #Modelo y tiempo de ejecución
        
        t = threading.Thread(target=ising_model, args=(M, T, N))
        threads.append(t)
        t.start()
        t.join()
        
        #Guardar parámetros de la simulación
        resultados[i, 0] = T
        resultados[i, 1] = M
        i += 1

    return resultados, threads


tiempo_0 = time.time()
#Ejecutar simulaciones
resultados, threads = simulaciones(lado_malla, temperaturas, pasos_monte)
    
tiempo_1 = time.time()
print('Tiempo de ejecución: ', tiempo_1 - tiempo_0)
print(resultados)



