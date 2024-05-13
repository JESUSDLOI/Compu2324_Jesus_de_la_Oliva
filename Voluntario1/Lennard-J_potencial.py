#Simulación del sistema solar.
#Unidades del sistema internacional.

import numpy as np
import time
from numba import jit

#Establecemos el tiempo inicial.
t0 = time.time()

#Establecemos los uncrementos del tiempo.
h = 0.001

#Número de iteraciones.
iteraciones = 100000

#Número de iteraciones que se saltan para guardar los datos.
skip = 10

#Valor sigma
sigma = 3.4

#Pedimos el número de partículas.
n = 20

#Tamaño de caja
l = 10

#Interespaciado entre las partículas.
s = 2

#Posición inicial de las partículas
@jit(nopython=True, fastmath=True, cache=True)
def posiciones_iniciales(n, l, s):
    posicion = np.zeros((n, 2)) + 1
    for i in range(n-1):
        x = posicion[i][0] + s
        if x > l:
            posicion[i+1] = np.array([1, posicion[i][1] + s])
        else:
            posicion[i+1] = np.array([posicion[i][0] + s, posicion[i][1]])
    return posicion


#Condiciónes de contorno periódicas.
@jit(nopython=True, fastmath=True, cache=True)
def contorno(posiciones, l):
    for i in range(n):
        x = posiciones[i][0]
        y = posiciones[i][1]
        if x > l:
            x = x % l
        if x < 0:
            x = x % l
        if y > l:
            y = y % l
        if y < 0:
            y = y % l
        posiciones[i][0] = x
        posiciones[i][1] = y
    return posiciones

#Función para calcular la distancia entre dos partículas.
@jit(nopython=True, fastmath=True, cache=True)
def distancia_condiciones(posicion, i, j, l):
    resta = np.zeros(2)
    mitad = l / 2
    
    resta[0] = posicion[i][0] - posicion[j][0]
    if abs(resta[0]) > mitad:
        resta[0] = -(l - abs(resta[0]))*np.sign(resta[0])
    
    resta[1] = posicion[i][1] - posicion[j][1]
    if abs(resta[1]) > mitad:
        resta[1] = -(l - abs(resta[1]))*np.sign(resta[1])
    
    distancia = np.sqrt(resta[0]**2 + resta[1]**2)
    
    return distancia, resta

#Definimos la función que nos da la acceleración en el tiempo t+h.
@jit(nopython=True, fastmath=True, cache=True)
def acel_i_th(n, posiciones, a_i, l, E_p_c, a_c):
    E_p = 0
    a_i = np.zeros((n, 2))
    for i in range(n):
        j = i+1
        while j < n:
            distancia, direccion = distancia_condiciones(posiciones, i, j, l)
            versor = direccion/distancia
            if distancia <= 3:
                aceleracion = (24/distancia**7)*(2/distancia**6 - 1) - a_c
                a_i[i] = a_i[i] + aceleracion*versor
                a_i[j] = a_i[j] - aceleracion*versor
                E_p += (4/distancia**6)*(1/distancia**6 - 1) - E_p_c + (distancia - 3)*a_c
            j = j+1
    return a_i, E_p


#Definimos la función w[i].
@jit(nopython=True, fastmath=True, cache=True)
def w_ih(n, velocidades, a_i, w_i, h):
    for i in range(n):
        w_i[i] = velocidades[i] + a_i[i]*(h/2)
    return w_i

#Definimos r(t+h) que nos da la nueva posición.
@jit(nopython=True, fastmath=True, cache=True)
def p_th(n, posiciones, w_i, h):
    for i in range(n):
        posiciones[i] = posiciones[i] + w_i[i]*h
    return posiciones

#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
@jit(nopython=True, fastmath=True, cache=True)
def velocidad_th(w_i, n, velocidades, a_i_th, h):
    E_c = 0
    for i in range(n):
        velocidades[i] = w_i[i] + a_i_th[i]*(h/2)
        E_c += energia_cinetica(velocidades[i])
    return velocidades, E_c

@jit(nopython=True, fastmath=True, cache=True)
def energia_cinetica(velocidades):
    energia_cinetica = (0.5)*(velocidades[0]**2 + velocidades[1]**2)
    return energia_cinetica


#Definimos las velocidades iniciales de las partículas.
velocidades = np.zeros((n, 2))
for i in range(n):
    velocidades[i][0] = 2*np.random.rand() - 1
    velocidades[i][1] = np.random.choice([-1, 1])*np.sqrt(1 - velocidades[i][0]**2)


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
a_i = np.zeros((n, 2))
#a_c es la aceleración de corte. Para ajustar el potencial de Lennard-Jones.
a_c = (2/3**7)*(2/3**6 - 1)
#E_p_c es la energía potencial de corte. Para ajustar el potencial de Lennard-Jones.
E_p_c = (4/3**6)*(1/3**6 - 1)
a_i, E_p = acel_i_th(n, posiciones, a_i, l, E_p_c, a_c)
w_i = np.zeros((n, 2))


#Calculamos la energía total del sistema.
energia = E_c + E_p


# Abrir tres archivos para guardar los datos de las posiciones, velocidades y energía.
file_posiciones = open('posiciones_part.dat', "w")
file_energia = open('energia_part.dat', "w")
file_energia_cinetica = open('energia_cinetica.dat', "w")
file_energia_potencial = open('energia_potencial.dat', "w")
file_velocidades = open('velocidades_part.dat', "w")


def guardar_datos(k, posiciones, energia, skip, E_c, E_p, mod_velocidades):
    if k % skip == 0:
        np.savetxt(file_posiciones, posiciones, delimiter=",")
        file_posiciones.write("\n")
        file_energia_cinetica.write(str(E_c) + "\n")
        file_energia_potencial.write(str(E_p) + "\n")
        file_energia.write(str(energia) + "\n")
        np.savetxt(file_velocidades, velocidades, delimiter=",")


# Realizamos el bucle para calcular las posiciones y velocidades de los planetas.
def simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c, skip):
    E_c_total = 0
    E_p_total = 0
    T = 0
    for k in range(iteraciones):

        guardar_datos(k, posiciones, energia, skip, E_c, E_p, velocidades)
        
        w_i = w_ih(n, velocidades, a_i, w_i, h)
        posiciones = p_th(n, posiciones, w_i, h)
        posiciones = contorno(posiciones, l)
        a_i, E_p = acel_i_th(n, posiciones, a_i, l, E_p_c, a_c)
        velocidades, E_c = velocidad_th(w_i, n, velocidades, a_i, h)
        energia = E_c + E_p

        E_c_total += E_c 
        E_p_total += E_p

    E_c_total = E_c_total/(iteraciones)
    E_p_total = E_p_total/(iteraciones)
    print("Energía cinética promedio: ", E_c_total)
    print("Energía potencial promedio: ", E_p_total)
    T = 2 * E_c_total/(iteraciones)
    print("Temperatura promedio: ", T)
    
#Ejecutamos la simulación.
simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c, skip)
    
# Cerrar los archivos
file_posiciones.close()
file_energia.close()

#Tiempo final.
t1 = time.time()

tiempo = t1 - t0

print("El tiempo de ejecución es: ", tiempo, "segundos")