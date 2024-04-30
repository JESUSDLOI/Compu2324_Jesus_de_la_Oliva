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
n = 20

#Tamaño de caja
l = 20

#Interespaciado entre las partículas.
s = 2

#Posición inicial de las partículas
@jit(nopython=True, fastmath=True, cache=True)
def posiciones_iniciales(n, l, s):
    posicion = np.zeros((n, 2))+ 1
    for i in range(n-1):
        x = posicion[i][0] + s
        if x > l:
            posicion[i+1] = [1, posicion[i][1] + s]
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
    
    resta[0] = posicion[i][0] - posicion[j][0]
    if abs(resta[0]) > l/2:
        resta[0] = (l - abs(resta[0])) 
    resta[0] = resta[0] * np.sign(resta[0])
    
    resta[1] = posicion[i][1] - posicion[j][1]
    if abs(resta[1]) > l/2:
        resta[1] = (l - abs(resta[1]))
    resta[1] = resta[1] * np.sign(resta[1])
    
    distancia = np.sqrt(resta[0]**2 + resta[1]**2)
    return distancia, resta

#Definimos la función a(t) a partir de la suma de fuerazas que se ejercen sobre cada partícula i.
#@jit(nopython=True, fastmath=True, cache=True)
def aceleracion_particulas(n, posicion, a_t, l, a_c):
    E_p = 0
    for i in range(n):
        j = i+1
        while j < n:
            distancia, direccion = distancia_condiciones(posicion, i, j, l)
            if distancia <= 3:
                aceleracion = (24/distancia**7)*(2/distancia**6 - 1) - a_c
                versor = direccion/distancia
                a_t[i] = a_t[i] + aceleracion * versor
                a_t[j] = a_t[j] - aceleracion * versor
                E_p += (8/distancia**6)*(1/distancia**6 - 1)    
            j = j+1             
    return a_t, E_p

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
#@jit(nopython=True, fastmath=True, cache=True)
def acel_i_th(n, posicion_th, a_i, a_c, l):
    E_p = 0
    for i in range(n):
        j = i+1
        while j < n:
            distancia, direccion = distancia_condiciones(posicion_th, i, j, l)
            if distancia <= 3:
                aceleracion = (24/distancia**7)*(2/distancia**6 - 1) - a_c
                versor = direccion/distancia
                a_i[i] = a_i[i] + aceleracion * versor
                a_i[j] = a_i[j] - aceleracion * versor
                E_p += (8/distancia**6)*(1/distancia**6 - 1)
            j = j+1
    return a_i, E_p


#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
@jit(nopython=True, fastmath=True, cache=True)
def velocidad_th(w_i, n, v_th, a_i_th, h):
    E_c = 0
    for i in range(n):
        v_th[i]= w_i[i] + a_i_th[i] *(h/2)
        E_c += energia_cinetica(v_th[i])
    return v_th, E_c

@jit(nopython=True, fastmath=True, cache=True)
def energia_cinetica(v):
    energia_cinetica = 0.5*np.sum(v**2)
    return energia_cinetica

#Definimos las velocidades iniciales de las partículas.
velocidades = np.random.uniform(-0.5, 0.5, (n, 2))*np.sqrt(12)

#Calculamos la velodad media.
v_media = np.mean(velocidades)

#Restamos la velocidad media a las velocidades iniciales, para que el sistema conserve la energía.
velocidades = velocidades - v_media

velocidades = np.zeros((n, 2))

@jit(nopython=True, fastmath=True, cache=True)
def energia_cinetica_inicial(velocidades, n):
    E_c = 0
    for i in range(n):
        E_c += energia_cinetica(velocidades[i])
    return E_c


#Inicializamos las variables que se utilizarán en el bucle.
posiciones = posiciones_iniciales(n, l, s)
posiciones = contorno(posiciones, l)
E_c = energia_cinetica_inicial(velocidades, n)
posicion_th = np.zeros((n, 2))
a_t = np.zeros((n, 2))
a_c = (24/3**7)*(2/3**13 - 1)
a_i, E_p = aceleracion_particulas(n, posiciones, a_t, l, a_c)
w_i = np.zeros((n, 2))
a_i_th = np.zeros((n, 2))
energia = 0




# Abrir tres archivos para guardar los datos de las posiciones, velocidades y energía.
file_posiciones = open('posiciones_part.dat', "w")
file_velocidades = open('velocidades_part.dat', "w")
file_aceleraciones = open('aceleraciones_part.dat', "w")
file_enegia = open('energia_part.dat', "w")
file_distancia = open('distancia_part.dat', "w")


def guardar_datos(k, n, posiciones, velocidades, energia):
    if k % 1 == 0:
        np.savetxt(file_posiciones, posiciones, delimiter=",")
        np.savetxt(file_velocidades, velocidades, delimiter=",")
        np.savetxt(file_aceleraciones, a_i, delimiter=",")
        file_posiciones.write("\n")
        file_velocidades.write("\n")
        file_aceleraciones.write("\n")
        file_enegia.write(str(energia) + "\n")

# Realizamos el bucle para calcular las posiciones y velocidades de los planetas.
def simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia):
    
    for k in range(iteraciones):

        guardar_datos(k, n, posiciones, velocidades, energia)

        w_i = w_ih(n, velocidades, a_i, w_i, h)
        posicion_th = p_th(n, posiciones, w_i, h)
        posicion_th = contorno(posicion_th, l)
        a_i_th, E_p = acel_i_th(n, posicion_th, a_i, a_c, l)
        v_th, E_c = velocidad_th(w_i, n, velocidades, a_i_th, h)
        posiciones = posicion_th
        velocidades = v_th
        a_i = a_i_th
        energia = E_c + E_p

#Ejecutamos la simulación.
simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia)
    
# Cerrar los archivos
file_posiciones.close()
file_velocidades.close()
file_aceleraciones.close()
file_enegia.close()

#Tiempo final.
t1 = time.time()

tiempo = t1 - t0

print("El tiempo de ejecución es: ", tiempo, "segundos")