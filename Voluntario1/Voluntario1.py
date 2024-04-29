#Simulación del sistema solar.
#Unidades del sistema internacional.

import numpy as np
import time
from numba import jit

#Establecemos el tiempo inicial.

t0 = time.time()

#Establecemos los uncrementos del tiempo.
h=0.00002

#Número de iteraciones.
iteraciones = 80000

#Valor sigma
sigma = 3.4

#Pedimos el número de partículas.
n = 10

#Tamaño de caja
l = 10

#Interespaciado entre las partículas.
s = 1.5

#Posición inicial de las partículas
@jit(nopython=True, fastmath=True, cache=True)
def posiciones_iniciales(n, l, s):
    posicion = np.zeros((n, 2))+ 1
    for i in range(n-1):
        x = posicion[i][0] + s
        if x > l:
            posicion[i+1] = [0, posicion[i][1] + s]
        else:
            posicion[i+1] = [posicion[i][0] + s, posicion[i][1]]
    return posicion

#Fuerza = -e[24/r-48/r]

#Condiciónes de contorno periódicas.
@jit(nopython=True, fastmath=True, cache=True)
def contorno(posicion_th, l):
    for i in range(n):
        x = posicion_th[i][0]
        y = posicion_th[i][1]
        if x > l:
            x = x % l
        if x < 0:
            x = x % l
        if y > l:
            y = y % l
        if y < 0:
            y = y % l
        posicion_th[i][0] = x
        posicion_th[i][1] = y
    return posicion_th

#Función para calcular la distancia entre dos partículas.
@jit(nopython=True, fastmath=True, cache=True)
def distancia_condiciones(posicion, i, j, l):
    resta = np.zeros(2)
    
    resta[0] = np.abs(posicion[i][0] - posicion[j][0])
    if resta[0] > l/2:
        resta[0] = np.abs(resta[0] - l)

    resta[1] = np.abs(posicion[i][1] - posicion[j][1])
    if resta[1] > l/2:
        resta[1] = np.abs(resta[1] - l)
    
    distancia = np.sqrt(resta[0]**2 + resta[1]**2)
    return distancia
    
#Definimos una función para calcular el versor de la distancia entre dos partículas.
@jit(nopython=True, fastmath=True, cache=True)
def versor_distancia(posicion, i, j, l):
    resta = np.zeros(2)
    
    resta[0] = posicion[i][0] - posicion[j][0]
    if abs(resta[0]) > l/2:
        resta[0] = resta[0] - l

    resta[1] = posicion[i][1] - posicion[j][1]
    if abs(resta[1]) > l/2:
        resta[1] = resta[1] - l

    return resta

#Definimos la función a(t) a partir de la suma de fuerazas que se ejercen sobre cada partícula i.
@jit(nopython=True, fastmath=True, cache=True)
def aceleracion_particulas(n, posicion, a_t):
    for i in range(n):
        for j in range(n):
            if i != j:
                distancia = distancia_condiciones(posicion, i, j, l)
                if distancia < 3 and distancia > 0.001:
                    versor = versor_distancia(posicion, i, j, l)
                    a_t[i] += (48/distancia**13 - 24*distancia**7)*versor/distancia               
    return a_t

#Definimos la función w[i].
@jit(nopython=True, fastmath=True, cache=True)
def w_ih(n, velocidades, a_i, w_i, h):
    for i in range(n):
        w_i[i] = velocidades[i] + a_i[i]*(h/2)
    return w_i

#Definimos r(t+h) que nos da la nueva posición.
@jit(nopython=True, fastmath=True, cache=True)
def p_th(n, posicion_th, w_i, h):
    for i in range(n):
        posicion_th[i] = posicion_th[i] + w_i[i]*h
    return posicion_th

#Definimos la función que nos da la acceleración en el tiempo t+h.
@jit(nopython=True, fastmath=True, cache=True)
def acel_i_th(n, posicion_th, a_i_th):
    for i in range(n):
        for j in range(n):
            if i != j:
                distancia = distancia_condiciones(posicion_th, i, j, l)
                if distancia < 3 and distancia > 0.001:
                    versor = versor_distancia(posicion_th, i, j, l)
                    a_i_th[i] += (48/distancia**13 - 24*distancia**7)*versor/distancia
    return a_i_th 


#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
@jit(nopython=True, fastmath=True, cache=True)
def velocidad_th(w_i, n, v_th, a_i_th, h):
    for i in range(n):
        v_th[i]= w_i[i] + a_i_th[i]*(h/2)
    return v_th


#Inicializamos las variables que se utilizarán en el bucle.
velocidades = np.round(np.random.uniform(-1, 1, (n, 2)), 2)*0.1
posiciones = posiciones_iniciales(n, l, s)
posicion_th = np.zeros((n, 2))
a_t = np.zeros((n, 2))
a_i = aceleracion_particulas(n, posiciones, a_t)
w_i = np.zeros((n, 2))
a_i_th = np.zeros((n, 2))

# Abrir tres archivos para guardar los datos de las posiciones, velocidades y aceleraciones
file_posiciones = open('posiciones_part.dat', "w")
file_velocidades = open('velocidades_part.dat', "w")


def guardar_datos(k, n, posiciones, velocidades):
    if k % 1 == 0:
        np.savetxt(file_posiciones, posiciones, delimiter=",")
        np.savetxt(file_velocidades, velocidades, delimiter=",")
        file_posiciones.write("\n")
        file_velocidades.write("\n")

# Realizamos el bucle para calcular las posiciones y velocidades de los planetas.
def simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones):
    
    for k in range(iteraciones):

        guardar_datos(k, n, posiciones, velocidades)

        w_i = w_ih(n, velocidades, a_i, w_i, h)
        posicion_th = p_th(n, posiciones, w_i, h)
        posicion_th = contorno(posicion_th, l)
        a_i_th = acel_i_th(n, posicion_th, a_i)
        v_th = velocidad_th(w_i, n, velocidades, a_i_th, h)
        posiciones = posicion_th
        velocidades = v_th
        a_i = a_i_th

#Ejecutamos la simulación.
simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones)
    
# Cerrar los archivos
file_posiciones.close()
file_velocidades.close()

#Tiempo final.
t1 = time.time()

tiempo = t1 - t0

print("El tiempo de ejecución es: ", tiempo, "segundos")