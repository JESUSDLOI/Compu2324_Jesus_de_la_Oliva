import numpy as np 
from numpy import random
from numba import jit
import matplotlib.pyplot as plt
import time

# ================================================================================

#ININICIAR VARIABLES

#Lado de la malla
lado_malla = np.linspace(20, 120, 5).astype(np.int8)

#Temperatura
temperaturas = np.linspace(1, 1, 5).astype(np.float32)

#Número de pasos_monte
pasos_monte = np.full(5, 40000).astype(np.int32)

#Calcular magnetización y energía
calcular_mag_ener = True

#Magnetización inicial
magnetizacion_inicial = 1
# ================================================================================


#Matriz aleatoria entre estado 1 y -1
@jit(nopython=True, fastmath=True)
def mtrz_aleatoria(M, magnetizacion_inicial):
    matriz = 2 * np.random.randint(0, 2, size=(M, M)).astype(np.int8) - 1
    if magnetizacion_inicial == 0:
        for i in range(M):
            for j in range(M):
                if j % 2 == 0:
                    matriz[i, j] = -1
                else:
                    matriz[i, j] = 1
                
    matriz[M-1] = np.ones(M).astype(np.int8)
    matriz[0] = np.ones(M).astype(np.int8)
    matriz[0] = -matriz[M-1]
    
    
    return matriz

#Condiciones de contorno periódicas
@jit(nopython=True, fastmath=True)
def cond_contorno(M, i, j):
    if i==M-1:      #
        arriba = M-2    #
        abajo = M-1  #
    elif i==0:      #
        arriba = 0      # Condiciones de contorno verticales
        abajo = 1    # 
    else:           # 
        arriba = i-1    # 
        abajo = i+1  #
        
    if j==M-1:       # 
        izquierda = j-1   # 
        derecha = 0    # 
    elif j==0:       # 
        izquierda = M-1   # Periodicidad horizontal
        derecha = 1    # 
    else:            # 
        izquierda = j-1   # 
        derecha = j+1  
        
    return izquierda, derecha, arriba, abajo


#Cálculo de la matriz
@jit(nopython=True, fastmath=True)
def calculo_matriz(matriz, M, T):
    cambio = True
    
    #Iteración sobre la matriz
    i = np.random.randint(1, M-2)
    j = np.random.randint(0, M)
    
    #Elección de la pareja
    eje = np.random.randint(0, 2)
    if eje == 0:
        i_pareja = i + 1
        j_pareja = j
    else:
        i_pareja = i
        if j == M-1:
            j_pareja = 0
        else:
            j_pareja = j + 1
        
    if matriz[i,j] == matriz[i_pareja, j_pareja]:
        cambio = False
    else:
        izquierda, derecha, arriba, abajo = cond_contorno(M, i, j)
        izquierda_pareja, derecha_pareja, arriba_pareja, abajo_pareja = cond_contorno(M, i_pareja, j_pareja)
        
        #Calculo de la variación de la energía
        #Cálculo de la energía y magnetización promedio
    
        #Cambio vertical
        if eje == 0:
            delta_E = 2*matriz[i,j]*(matriz[i, derecha] + matriz[i, izquierda] + matriz[arriba, j] - matriz[i_pareja, derecha_pareja] - matriz[i_pareja, izquierda_pareja] - matriz[abajo_pareja, j_pareja])
        #Cambio horizontal
        else:
            delta_E = 2*matriz[i,j]*(matriz[arriba, j] + matriz[abajo, j] + matriz[i, izquierda] - matriz[arriba_pareja, j_pareja] - matriz[abajo_pareja, j_pareja] - matriz[i_pareja, derecha_pareja])
       
        #Probabilidad de cambio
        p_0 = np.exp(-delta_E/T)

        #Evaluar probabilidad de cambio
        if p_0 > 1:
            p = 1
        else:
            p = p_0

        #Número aleatorio entre  para comparar con la probabilidad
        r = np.random.rand()

        #Comparar probabilidad para cambiar el spin
        if r < p:
            matriz[i, j], matriz[i_pareja, j_pareja] = matriz[i_pareja, j_pareja], matriz[i, j]
                     
    return matriz, cambio

#Secuencia de Ising
@jit(nopython=True, fastmath=True)
def secuencia_isin(M, T, matriz, n, magnt_prom, E, m_cuadrado, calcular_mag_ener):
    
    #Matriz de Ising
    matriz, cambio = calculo_matriz(matriz, M, T)

    magnt_prom = 0
    magnt_prom_superior = 0
    magnt_prom_inferior = 0 
    E = 0
    E_sup = 0
    E_inf = 0

    if calcular_mag_ener == True:
        r = 0
        if n % 100 == 0:
            #Cálculo de la energía y magnetización promedio
            i = 0
            j = 0
            while i < M:
                j = 0
                while j < M:
                    izquierda, derecha, arriba, abajo = cond_contorno(M, i, j)
                    if i < M/2:
                        magnt_prom_superior += matriz[i, j]
                        E_sup += matriz[i,j]*(matriz[(derecha),j] + matriz[i,(abajo)] + matriz[(izquierda),j] + matriz[i,(arriba)])
                        r += 1
                    else:
                        magnt_prom_inferior += matriz[i, j]
                        E_inf += matriz[i,j]*(matriz[(derecha),j] + matriz[i,(abajo)] + matriz[(izquierda),j] + matriz[i,(arriba)])
                    j += 1
                i += 1
            E_sup = -E_sup/2 
            E_inf = -E_inf/2   
            E = (E_sup + E_inf)/(m_cuadrado)
            magnt_prom_superior = magnt_prom_superior/(r)
            magnt_prom_inferior = magnt_prom_inferior/(m_cuadrado - r)  
            magnt_prom = magnt_prom_superior + magnt_prom_inferior
        return magnt_prom, E, matriz, cambio, magnt_prom_superior, magnt_prom_inferior, E_sup, E_inf
    else:
        return magnt_prom, E, matriz, cambio, magnt_prom_superior, magnt_prom_inferior, E_sup, E_inf

#Matriz de Ising
def ising_model(M, T, N, calcular_mag_ener, magnetizacion_inicial):
    #Variables
    k= 0
    n = 0
    
    magnetizaciones = []    
    energias = []
    magnetizaciones_superior = []
    magnetizaciones_inferior = []
    energias_superior = []
    energias_inferior = []
    magnt_prom = 0
    magnt_prom_superior = 0
    magnt_prom_inferior = 0 
    E = 0
    E_sup = 0
    E_inf = 0


    #Matriz de Ising
    matriz = mtrz_aleatoria(M, magnetizacion_inicial)
    m_cuadrado = M**2

    #Archivo de datos
    with open('ising_data_tem_{0:.2f}_malla_{1}.dat'.format(T, M), 'w') as file:
        while n < N:
            k = 0
            while k < (m_cuadrado):
                #Resultados
                if calcular_mag_ener == True:
                    magnt_prom, E, matriz, cambio, magnt_prom_superior, magnt_prom_inferior, E_sup, E_inf = secuencia_isin(M, T, matriz, n, magnt_prom, E, m_cuadrado, calcular_mag_ener)
                    if n % 100 == 0:    
                        magnetizaciones.append(magnt_prom)
                        energias.append(E)
                        magnetizaciones_inferior.append(magnt_prom_inferior)
                        magnetizaciones_superior.append(magnt_prom_superior)
                        energias_superior.append(E_sup)
                        energias_inferior.append(E_inf)
                else:
                    magnt_prom, E, matriz, cambio, magnt_prom_superior, magnt_prom_inferior, E_sup, E_inf = secuencia_isin(M, T, matriz, n, magnt_prom, E, m_cuadrado, calcular_mag_ener)
                k += 1
                    
            if n % 10 == 0:
                file.write('\n')
                np.savetxt(file, matriz, fmt='%d', delimiter=',')                        
            n += 1
            
    return energias, magnetizaciones, magnetizaciones_superior, magnetizaciones_inferior, E_sup, E_inf


def graficar(magn, en, magnetizaciones_superior, magnetizaciones_inferior, E_sup, E_inf, calcular_mag_ener, T, M):
    if calcular_mag_ener == True:
        
        plt.rcParams['font.size'] = 20
        
        # Crear una figura
        plt.figure(figsize=(20, 8))
        # Crear la primera gráfica
        plt.subplot(1, 2, 1)  # 2 filas, 1 columna, primer gráfico

        # Graficar las magnetizaciones
        plt.plot(magnetizaciones_superior, label='Magnetizaciones Superior')
        plt.plot(magnetizaciones_inferior, label='Magnetizaciones Inferior')
        plt.plot(magn, label='Magnetizaciones')
        plt.legend()

        # Añadir títulos y etiquetas
        plt.title('Magnetizaciones Superior, Inferior y total para T = {0:.2f} y M = {1}'. format(T, M))
        plt.xlabel('Pasos')
        plt.ylabel('Magnetización')

        # Crear la segunda gráfica
        plt.subplot(1, 2, 2)  # 2 filas, 1 columna, segundo gráfico

        # Graficar las energías
        plt.plot(E_sup, label='Energía Superior')
        plt.plot(E_inf, label='Energía Inferior')
        plt.plot(en, label='Energía')

        # Añadir títulos y etiquetas
        plt.title('Energías Superior e Inferior para T = {0:.2f} y M = {1}'. format(T, M))
        plt.xlabel('Pasos')
        plt.ylabel('Energía')

        # Mostrar la leyenda
        plt.legend()

        # Mostrar la gráfica
        plt.savefig('Simulación ising kawasaki_T_{0:.2f}_M_{1}.png'.format(T, M))
                
        


#Simulaciones de Monte Carlo distintas temperaturas y mallas
def simulaciones(lado_malla, temperaturas, pasos_monte, calcular_mag_ener, magnetizacion_inicial):

    #Cantidad de archivos
    C = len(temperaturas)
    resultados = np.zeros((C, 2))
    
    for i in range(C):
        #Temperatura y lado de la malla
        T = temperaturas[i]
        M = lado_malla[i]
        N = pasos_monte[i]

        #Modelo y tiempo de ejecución
        tiempo_0 = time.time()
        en, magn, magnetizaciones_superior, magnetizaciones_inferior, E_sup, E_inf = ising_model(M, T, N, calcular_mag_ener, magnetizacion_inicial)
        tiempo_1 = time.time()

        tiempo = tiempo_1 - tiempo_0
        if calcular_mag_ener == True:
            magn = np.array(magn)
            en = np.array(en)
            magnetizaciones_inferior = np.array(magnetizaciones_inferior)
            magnetizaciones_superior = np.array(magnetizaciones_superior)
            E_inf = np.array(E_inf)
            E_sup = np.array(E_sup)
            energia_cuadra = en**2
            promedio_mag = np.mean(magn)
            media_eners = np.mean(en)
            promedio_energia = media_eners/(M**2)
            calor_especif = (np.mean(energia_cuadra) - media_eners**2)/(T*M)**2

            open('resultados_{0:.2f}_{1}.dat'.format(T, M) , 'w').write(f'Magnetizacion prmedio {promedio_mag} \n Energia promedio {promedio_energia} \n Calor especifico {calor_especif} \n Tiempo {tiempo}\n')
            graficar(magn, en, magnetizaciones_superior, magnetizaciones_inferior, E_sup, E_inf, calcular_mag_ener, T, M)
            
        #Guardar parámetros de la simulación
        resultados[i, 0] = T
        resultados[i, 1] = M
        
        print('Simulación terminada para T = {0:.2f} y M = {1}'.format(T, M))
        print('Tiempo de ejecución: {0:.2f} s'.format(tiempo))
        
    return magn, en, magnetizaciones_superior, magnetizaciones_inferior, E_sup, E_inf
    
magn, en, magnetizaciones_superior, magnetizaciones_inferior, E_sup, E_inf = simulaciones(lado_malla, temperaturas, pasos_monte, calcular_mag_ener, magnetizacion_inicial)

print('Simulaciones terminadas')



                                  

