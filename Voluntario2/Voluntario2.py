import numpy as np 
from numpy import random
from numba import jit
import time

# ================================================================================

#ININICIAR VARIABLES

#Lado de la malla
lado_malla = np.full(3, 10).astype(np.int8)

#Temperatura
temperaturas = np.linspace(0.5, 5, 3).astype(np.float32)

#Número de pasos_monte
pasos_monte = np.full(3, 1000000).astype(np.int32)

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
        izquierda = i
    if i == M - 1:
        derecha = 0
    else:
        derecha = i
    if j == 0:
        arriba = M - 1
    else:
        arriba = j
    if j == M - 1:
        abajo = 0
    else:
        abajo = j
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
def secuencia_isin(M, T, matriz, n, magnt_prom, E, m_cuadrado):
    
    #Variables
    i, j, delta_E = calculo_matriz(matriz, M)
    
    #Evaluar probabilidad de cambio
    p_r = np.exp(-2*delta_E/T)

    if p_r > 1:
        p = 1
    else:
        p = p_r

    #Número aleatorio entre  para comparar con la probabilidad
    r = np.random.uniform(0, 1)

    #Comparar probabilidad para cambiar el spin
    if r < p:
        matriz[i,j] = -matriz[i,j]  

    if n % 100 == 0:
        #Cálculo de la energía y magnetización promedio
        i = 0
        j = 0
        while i < M:
            while j < M:
                izquierda, derecha, arriba, abajo = cond_contorno(M, i, j)
                magnt_prom += matriz[i,j]
                E += matriz[i,j]*(matriz[(derecha),j] + matriz[i,(abajo)] + matriz[(izquierda),j] + matriz[i,(arriba)])
                j += 1
            i += 1
        E = -E/4
        magnt_prom = magnt_prom/(m_cuadrado)

    return magnt_prom, E

#Matriz de Ising
@jit(nopython=True, fastmath=True, cache=True)
def ising_model(M, T, N):
    #Variables
    k= 0
    n = 0
    magnetizaciones = []    
    energias = []
    magnt_prom = 0
    E = 0

    #Matriz de Ising
    matriz = mtrz_aleatoria(M)
    m_cuadrado = M**2

    #Archivo de datos
    while n < N:
        while k < (m_cuadrado):
            
            #Resultados
            magnt_prom, E = secuencia_isin(M, T, matriz, n, magnt_prom, E, m_cuadrado)
            magnetizaciones.append(magnt_prom)
            energias.append(E)
            k += 1
        n += 1
   
    return energias, magnetizaciones





#Simulaciones de Monte Carlo distintas temperaturas y mallas
def simulaciones(lado_malla, temperaturas, pasos_monte):

    #Cantidad de archivos
    C = len(temperaturas)
    resultados = np.zeros((C, 2))
    
    for i in range(C):
        #Temperatura y lado de la malla
        T = temperaturas[i]
        M = lado_malla[i]
        N = pasos_monte[i]
        en = 0
        magn = 0

        #Modelo y tiempo de ejecución
        tiempo_0 = time.time()
        en, magn  = ising_model(M, T, N)
        tiempo_1 = time.time()

        tiempo = tiempo_1 - tiempo_0

        energia_cuadra = np.square(np.array(en))
        promedio_mag = np.mean(magn)
        media_eners = np.mean(en)
        promedio_energia = media_eners/(2*M)
        calor_especif = (np.mean(energia_cuadra) - media_eners**2)/(T*M**2)

        open('resultados_'+str(T)+'_'+str(M)+'.dat' , 'w').write(f'Magnetizacion prmedio {promedio_mag} \n Energia promedio {promedio_energia} \n Calor especifico {calor_especif} \n Tiempo {tiempo}\n')

        #Guardar parámetros de la simulación
        resultados[i, 0] = T
        resultados[i, 1] = M
    
    
simulaciones(lado_malla, temperaturas, pasos_monte)



                                  

