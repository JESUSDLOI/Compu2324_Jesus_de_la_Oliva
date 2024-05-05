#Simulación del sistema solar.
#Unidades del sistema internacional.

import numpy as np
import time
from numba import jit
from decimal import Decimal, getcontext

#Establecemos el tiempo inicial.
t0 = time.time()

#Establecemos la precisión de los decimales.
getcontext().prec = 30

#Establecemos los uncrementos del tiempo.
h = Decimal('0.001')

#Número de iteraciones.
iteraciones = 10000

#Número de iteraciones que se saltan para guardar los datos.
skip = 1

#Valor sigma
sigma = 3.4

#Pedimos el número de partículas.
n = 2

#Tamaño de caja
l = Decimal('10')

#Interespaciado entre las partículas.
s = Decimal('1.3')

#Posición inicial de las partículas
#@jit(nopython=True, fastmath=True, cache=True)
def posiciones_iniciales(n, l, s):
    posicion = np.full((n, 2), Decimal('0')) + Decimal('1')
    for i in range(n-1):
        x = posicion[i][0] + s
        if x > l:
            posicion[i+1] = [Decimal('1'), posicion[i][1] + s]
        else:
            posicion[i+1] = [posicion[i][0] + s, posicion[i][1]]
    return posicion


#Condiciónes de contorno periódicas.
#@jit(nopython=True, fastmath=True, cache=True)
def contorno(posiciones, l):
    for i in range(n):
        x = Decimal(str(posiciones[i][0]))
        y = Decimal(str(posiciones[i][1]))
        if x > l:
            x = x % Decimal('1')
        if x < 0:
            x = x % Decimal('1')
        if y > l:
            y = y % Decimal('1')
        if y < 0:
            y = y % Decimal('1')
        posiciones[i][0] = x
        posiciones[i][1] = y
    return posiciones

#Función para calcular la distancia entre dos partículas.
#@jit(nopython=True, fastmath=True, cache=True)
def distancia_condiciones(posicion, i, j, l):
    resta = np.full((2), Decimal('0'))
    mitad = l / Decimal('2')
    
    resta[0] = posicion[i][0] - posicion[j][0]
    if abs(resta[0]) > mitad:
        resta[0] = -(l - abs(resta[0]))*np.sign(resta[0])
    
    resta[1] = posicion[i][1] - posicion[j][1]
    if abs(resta[1]) > mitad:
        resta[1] = -(l - abs(resta[1]))*np.sign(resta[1])
    
    distancia = resta[0]**Decimal('2') + resta[1]**Decimal('2')
    distancia = distancia.sqrt()
    
    return distancia, resta

#Definimos la función que nos da la acceleración en el tiempo t+h.
#@jit(nopython=True, fastmath=True)
def acel_i_th(n, posiciones, a_i, l, E_p_c, a_c):
    E_p = Decimal('0')
    for i in range(n):
        j = i+1
        while j < n:
            distancia, direccion = distancia_condiciones(posiciones, i, j, l)
            versor = direccion/distancia
            if distancia <= 3:
                aceleracion = (Decimal('24')/distancia**Decimal('7'))*(Decimal('2')/distancia**Decimal('6') - Decimal('1')) - a_c
                a_i[i] = a_i[i] + aceleracion*versor
                a_i[j] = a_i[j] - aceleracion*versor
                E_p += (Decimal('4')/distancia**Decimal('6'))*(Decimal('1')/distancia**Decimal('6') - Decimal('1')) - E_p_c + (distancia - Decimal('3'))*a_c
            j = j+1
    return a_i, E_p


#Definimos la función w[i].
#@jit(nopython=True, fastmath=True, cache=True)
def w_ih(n, velocidades, a_i, w_i, h):
    for i in range(n):
        w_i[i] = velocidades[i] + a_i[i]*(h/Decimal('2'))
    return w_i

#Definimos r(t+h) que nos da la nueva posición.
#@jit(nopython=True, fastmath=True)
def p_th(n, posiciones, w_i, h):
    for i in range(n):
        posiciones[i] = posiciones[i] + w_i[i]*h
    return posiciones

#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
#@jit(nopython=True, fastmath=True)
def velocidad_th(w_i, n, velocidades, a_i_th, h):
    E_c = Decimal('0')
    for i in range(n):
        velocidades[i] = w_i[i] + a_i_th[i]*(h/Decimal('2'))
        E_c += energia_cinetica(velocidades[i])
    return velocidades, E_c

#@jit(nopython=True, fastmath=True)
def energia_cinetica(velocidades):
    energia_cinetica = Decimal('0.5')*(velocidades[0]**Decimal('2') + velocidades[1]**Decimal('2'))
    return energia_cinetica


'''
#Definimos las velocidades iniciales de las partículas.
velocidades = np.random.uniform(-0.5, 0.5, (n, 2))*np.sqrt(12)

#Calculamos la velodad media.
v_media = np.mean(velocidades)

#Restamos la velocidad media a las velocidades iniciales, para que el sistema conserve la energía.
velocidades = velocidades - v_media
'''

velocidades = np.full((n, 2), Decimal('0'))

#@jit(nopython=True, fastmath=True)
def energia_cinetica_inicial(velocidades, n):
    E_c = Decimal('0')
    for i in range(n):
        E_c += energia_cinetica(velocidades[i])
    return E_c


#Inicializamos las variables que se utilizarán en el bucle.
posiciones = posiciones_iniciales(n, l, s)
posiciones = contorno(posiciones, l)
E_c = energia_cinetica_inicial(velocidades, n)
a_i = np.full((n, 2), Decimal('0'))
#a_c es la aceleración de corte. Para ajustar el potencial de Lennard-Jones.
a_c = (Decimal('24')/Decimal('3')**Decimal('7'))*(Decimal('2')/Decimal('3')**Decimal('6') - Decimal('1'))
#E_p_c es la energía potencial de corte. Para ajustar el potencial de Lennard-Jones.
E_p_c = (Decimal('4')/Decimal('3')**Decimal('6'))*(Decimal('1')/Decimal('3')**Decimal('6') - Decimal('1'))
a_i, E_p = acel_i_th(n, posiciones, a_i, l, E_p_c, a_c)
w_i = np.full((n, 2), Decimal('0'))


#Calculamos la energía total del sistema.
energia = E_c + E_p


# Abrir tres archivos para guardar los datos de las posiciones, velocidades y energía.
file_posiciones = open('posiciones_part.dat', "w")
file_velocidades = open('velocidades_part.dat', "w")
file_aceleraciones = open('aceleraciones_part.dat', "w")
file_enegia = open('energia_part.dat', "w")


def guardar_datos(k, n, posiciones, velocidades, energia, skip):
    if k % skip == 0:
        np.savetxt(file_posiciones, posiciones, delimiter=",")
        np.savetxt(file_velocidades, velocidades, delimiter=",")
        np.savetxt(file_aceleraciones, a_i, delimiter=",")
        file_posiciones.write("\n")
        file_velocidades.write("\n")
        file_aceleraciones.write("\n")
        file_enegia.write(str(energia) + "\n")

# Realizamos el bucle para calcular las posiciones y velocidades de los planetas.
def simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c, skip):
    
    for k in range(iteraciones):

        guardar_datos(k, n, posiciones, velocidades, energia, skip)
        
        w_i = w_ih(n, velocidades, a_i, w_i, h)
        posiciones = p_th(n, posiciones, w_i, h)
        posiciones = contorno(posiciones, l)
        a_i, E_p = acel_i_th(n, posiciones, a_i, l, E_p_c, a_c)
        velocidades, E_c = velocidad_th(w_i, n, velocidades, a_i, h)
        energia = E_c + E_p

#Ejecutamos la simulación.
simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c, skip)
    
# Cerrar los archivos
file_posiciones.close()
file_velocidades.close()
file_aceleraciones.close()
file_enegia.close()

#Tiempo final.
t1 = time.time()

tiempo = t1 - t0

print(Decimal('1')/3)

print("El tiempo de ejecución es: ", tiempo, "segundos")