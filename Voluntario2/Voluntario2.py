import numpy as np 
from numpy import random
from numba import jit
import time

# ================================================================================

#ININICIAR VARIABLES

#Lado de la malla
lado_malla = np.full(2, 10).astype(np.int8)

#Temperatura
temperaturas = np.linspace(0.5, 5, 2).astype(np.float32)

#Número de pasos_monte
pasos_monte = np.full(2, 100000).astype(np.int32)

# ================================================================================


#Matriz aleatoria entre estado 1 y -1
@jit(nopython=True, fastmath=True)
def mtrz_aleatoria(M):
    matriz = 2 * np.random.randint(0, 2, size=(M, M)).astype(np.int8) - 1
    matriz[0] = np.ones(M).astype(np.int8)
    matriz[M-1] = np.ones(M).astype(np.int8)
    matriz[M-1] = -matriz[M-1]
    return matriz

#Condiciones de contorno periódicas
@jit(nopython=True, fastmath=True)
def cond_contorno(M, i, j):
    if j == 0:
        izquierda = M - 1
        derecha = 1
    elif j == (M - 1):
        derecha = 0
        izquierda = j - 1
    else:
        derecha = j + 1
        izquierda = j - 1
        
    if i == 0:
        arriba = i
        abajo = i + 1
    elif i == (M - 1):
        abajo = i
        arriba = i - 1
    else:
        arriba = i - 1
        abajo = i + 1
        
    return izquierda, derecha, arriba, abajo

#Cálculo de la matriz
@jit(nopython=True, fastmath=True)
def calculo_matriz(matriz, M):
    cambio = True
    
    #Iteración sobre la matriz
    i = np.random.randint(1, M-1)
    j = np.random.randint(0, M)
    
    #Elección de la pareja
    eje = np.random.randint(0, 2)
    if eje == 0:
        i_pareja = i + 2*np.random.randint(0, 2) - 1
        j_pareja = j
    else:
        j_pareja = j + 2*np.random.randint(0, 2) - 1
        if j_pareja == -1:
            j_pareja = M - 1
        elif j_pareja == M:
            j_pareja = 0
        i_pareja = i

    if i_pareja == 0 or i_pareja == (M - 1) or matriz[i,j] == matriz[i_pareja, j_pareja]:
        cambio = False
    else:
        izquierda, derecha, arriba, abajo = cond_contorno(M, i, j)
        izquierda_pareja, derecha_pareja, arriba_pareja, abajo_pareja = cond_contorno(M, i_pareja, j_pareja)
        #Calculo de la variación de la energía
        if eje == 0:
            delta_E = 2*matriz[i,j]*(matriz[i, derecha] + matriz[i, izquierda] + matriz[arriba, j] - matriz[i_pareja, derecha_pareja] - matriz[i_pareja, izquierda_pareja] - matriz[abajo_pareja, j_pareja])
        else:
            delta_E = 2*matriz[i,j]*(matriz[arriba, j] + matriz[abajo, j] + matriz[i, izquierda] - matriz[arriba_pareja, j_pareja] - matriz[abajo_pareja, j_pareja] - matriz[i_pareja, derecha_pareja])
        
                  
    return i, j, i_pareja, j_pareja, delta_E, cambio

#Secuencia de Ising
@jit(nopython=True, fastmath=True)
def secuencia_isin(M, T, matriz, n, magnt_prom, E, m_cuadrado):
    
    #Variables
    i, j, i_pareja, j_pareja ,delta_E, cambio = calculo_matriz(matriz, M)
    
    if cambio == False:
        pass
    else:
        #Probabilidad de cambio
        p_0 = np.exp(-delta_E/T)

        #Evaluar probabilidad de cambio
        p = min(1, p_0)

        #Número aleatorio entre  para comparar con la probabilidad
        r = np.random.uniform(0, 1)

        #Comparar probabilidad para cambiar el spin
        if r < p:
            matriz[i, j] = -matriz[i, j]  
            matriz[i_pareja, j_pareja] = -matriz[i_pareja, j_pareja]

    #if n % 100 == 0:
    #    #Cálculo de la energía y magnetización promedio
    #    i = 0
    #    j = 0
    #    while i < M:
    #        while j < M:
    #            izquierda, derecha, arriba, abajo = cond_contorno(M, i, j)
    #            magnt_prom += matriz[i,j]
    #            E += matriz[i,j]*(matriz[(derecha),j] + matriz[i,(abajo)] + matriz[(izquierda),j] + matriz[i,(arriba)])
    #            j += 1
    #        i += 1
    #    E = -E/4
    #    magnt_prom = magnt_prom/(m_cuadrado)

    return magnt_prom, E, matriz

#Matriz de Ising
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
    with open('ising_data_tem_{0:.2f}_malla_{1}.dat'.format(T, M), 'w') as file:
        while n < N:
            k = 0
            while k < (m_cuadrado):
                
                #Resultados
                magnt_prom, E, matriz = secuencia_isin(M, T, matriz, n, magnt_prom, E, m_cuadrado)
                magnetizaciones.append(magnt_prom)
                energias.append(E)
                k += 1
                if n % 1000 == 0:
                    file.write('\n')
                    np.savetxt(file, matriz, fmt='%d', delimiter=',') 
                        
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
        
        print('Simulación terminada para T = {0:.2f} y M = {1}'.format(T, M))
        print('Tiempo de ejecución: {0:.2f} s'.format(tiempo))
    
    
simulaciones(lado_malla, temperaturas, pasos_monte)
print('Simulaciones terminadas')



                                  

