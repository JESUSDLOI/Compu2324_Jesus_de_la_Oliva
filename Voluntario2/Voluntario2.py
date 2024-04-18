import numpy as np 
from numpy import random
from numba import jit
import time

# ================================================================================

#ININICIAR VARIABLES

#Lado de la malla
lado_malla = (10, 15, 16)

#Temperatura
temperaturas = (1, 1, 1.5)

#Número de pasos_monte
pasos_monte = (100, 10000, 10000)

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
def secuencia_isin(M, T, matriz, n, magnt_prom, E):
    
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

    if n % 100 == 0:
        #Cálculo de la magnetización
        magnt_prom = np.sum(submatriz)
        #Cálculo de la energía
        i = 1
        j = 1
        while i < M+1:
            while j < M+1:
                E += matriz[i,j]*(matriz[(i+1),j] + matriz[i,(j+1)] + matriz[(i-1),j] + matriz[i,(j-1)])
                j += 1
            i += 1
        E = -E/4

    return magnt_prom, E

#Matriz de Ising
def ising_model(M, T, N):
    #Variables
    magnt_prom = np.array([])
    E = np.array([])
    prob = np.zeros([])
    k= 0
    Z = 0
    n = 0
    

    #Matriz de Ising
    matriz = mtrz_aleatoria(M)

    #Condiciones de contorno
    matriz = cond_contorno(matriz, M)

    #Archivo de datos
    while n < N:
        while k < (M**2):
            #Matriz resultado
            magnt_prom, E = secuencia_isin(M, T, matriz, n, magnt_prom, E)
            k += 1
        n += 1
        if n % 100 == 0:
            prob[l] = np.exp(-E/T)
            l += 1
    return Z, 





#Simulaciones de Monte Carlo distintas temperaturas y mallas
def simulaciones(lado_malla, temperaturas, pasos_monte):
    #Cantidad de archivos
    C = len(temperaturas)
    resultados = np.zeros((C, 3))

    for i in range(C):
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

    return resultados

print(simulaciones(lado_malla, temperaturas, pasos_monte))