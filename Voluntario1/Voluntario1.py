#Simulación del sistema solar.
#Unidades del sistema internacional.

import numpy as np
import time
from numba import jit

#Establecemos el tiempo inicial.

t0 = time.time()

#Establecemos los uncrementos del tiempo.
h=0.001

#Número de iteraciones.
iteraciones = 1000

#Valor sigma
sigma = 3.4

#Pedimos el número de partículas.
n = 20

#Tamaño de caja
l = 10

#Interespaciado entre las partículas.
s = 2

#Posición inicial de las partículas
@jit(nopython=True, fastmath=True)
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
@jit(nopython=True, fastmath=True)
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
@jit(nopython=True, fastmath=True)
def distancia_condiciones(posicion, i, j, l):
    resta = np.zeros(2)
    mitad = l/2
    
    resta[0] = posicion[i][0] - posicion[j][0]
    if abs(resta[0]) > mitad:
        resta[0] = -(l - abs(resta[0]))*np.sign(resta[0])
    
    resta[1] = posicion[i][1] - posicion[j][1]
    if abs(resta[1]) > mitad:
        resta[1] = -(l - abs(resta[1]))*np.sign(resta[1])
    
    distancia = round(np.sqrt(resta[0]**2 + resta[1]**2), 5)
    return distancia, resta

#Definimos la función a(t) a partir de la suma de fuerazas que se ejercen sobre cada partícula i.
@jit(nopython=True, fastmath=True)
def aceleracion_particulas(n, posicion, a_t, l, E_p_c, a_c):
    E_p = 0
    for i in range(n):
        j = i+1
        while j < n:
            distancia, direccion = distancia_condiciones(posicion, i, j, l)
            if distancia <= 3:
                aceleracion = (24/distancia**7)*((2/distancia**6) - 1) - a_c
                versor = direccion/distancia
                a_t[i] = a_t[i] + (aceleracion * versor)
                a_t[j] = a_t[j] - (aceleracion * versor)
                E_p += ((4/distancia**6)*(1/distancia**6 - 1)) - E_p_c + (distancia - 3)*a_c
            j = j+1             
    return a_t, E_p

#Definimos la función w[i].
@jit(nopython=True, fastmath=True)
def w_ih(n, velocidades, a_i, w_i, h):
    for i in range(n):
        w_i[i] = velocidades[i] + a_i[i]*(h/2)
    return w_i

#Definimos r(t+h) que nos da la nueva posición.
@jit(nopython=True, fastmath=True)
def p_th(n, posiciones, w_i, h):
    for i in range(n):
        posiciones[i] = posiciones[i] + w_i[i]*h
    return posiciones

#Definimos la función que nos da la acceleración en el tiempo t+h.
@jit(nopython=True, fastmath=True)
def acel_i_th(n, posicion_th, a_i, l, E_p_c, a_c):
    E_p = 0
    for i in range(n):
        j = i+1
        while j < n:
            distancia, direccion = distancia_condiciones(posicion_th, i, j, l)
            versor = direccion/distancia
            if distancia <= 3:
                aceleracion = (24/distancia**7)*(2/distancia**6 - 1) - a_c
                a_i[i] = a_i[i] + aceleracion*versor
                a_i[j] = a_i[j] - aceleracion*versor
                E_p += (4/distancia**6)*(1/distancia**6 - 1) - E_p_c + (distancia - 3)*a_c
            j = j+1
            #file_distancia.write(str(distancia) + "\n")
    return a_i, E_p


#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
@jit(nopython=True, fastmath=True)
def velocidad_th(w_i, n, velocidades, a_i_th, h):
    E_c = 0
    for i in range(n):
        velocidades[i] = w_i[i] + a_i_th[i]*(h/2)
        E_c += energia_cinetica(velocidades[i])
    return velocidades, E_c

@jit(nopython=True, fastmath=True)
def energia_cinetica(velocidades):
    energia_cinetica = 0.5*(velocidades[0]**2 + velocidades[1]**2)
    return energia_cinetica

#Definimos las velocidades iniciales de las partículas.
velocidades = np.random.uniform(-0.5, 0.5, (n, 2))*np.sqrt(12)

#Calculamos la velodad media.
v_media = np.mean(velocidades)

#Restamos la velocidad media a las velocidades iniciales, para que el sistema conserve la energía.
velocidades = velocidades - v_media

velocidades = np.zeros((n, 2))

@jit(nopython=True, fastmath=True)
def energia_cinetica_inicial(velocidades, n):
    E_c = 0
    for i in range(n):
        E_c += energia_cinetica(velocidades[i])
    return E_c


#Inicializamos las variables que se utilizarán en el bucle.
posiciones = posiciones_iniciales(n, l, s)
posiciones = contorno(posiciones, l)
E_c = energia_cinetica_inicial(velocidades, n)
a_t = np.zeros((n, 2))
#a_c es la aceleración de corte. Para ajustar el potencial de Lennard-Jones.
a_c = (24/3**7)*(2/3**6 - 1)
#E_p_c es la energía potencial de corte. Para ajustar el potencial de Lennard-Jones.
E_p_c = (4/3**6)*(1/3**6 - 1)
a_i, E_p = aceleracion_particulas(n, posiciones, a_t, l, E_p_c, a_c)
w_i = np.zeros((n, 2))


#Calculamos la energía total del sistema.
energia = E_c + E_p

print("La energía cinética es: ", E_c)
print("La energía potencial es: ", E_p)
print("La energía total es: ", energia)

# Abrir tres archivos para guardar los datos de las posiciones, velocidades y energía.
file_posiciones = open('posiciones_part.dat', "w")
file_velocidades = open('velocidades_part.dat', "w")
file_aceleraciones = open('aceleraciones_part.dat', "w")
file_enegia = open('energia_part.dat', "w")

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
def simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c):
    
    for k in range(iteraciones):

        guardar_datos(k, n, posiciones, velocidades, energia)
        
        w_i = w_ih(n, velocidades, a_i, w_i, h)
        posiciones = p_th(n, posiciones, w_i, h)
        posiciones = contorno(posiciones, l)
        a_i, E_p = acel_i_th(n, posiciones, a_i, l, E_p_c, a_c)
        velocidades, E_c = velocidad_th(w_i, n, velocidades, a_i, h)
        energia = E_c + E_p
        print("La energía cinética es: ", E_c)
        print("La energía potencial es: ", E_p)
        print("La energía total es: ", energia)

#Ejecutamos la simulación.
simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c)
    
# Cerrar los archivos
file_posiciones.close()
file_velocidades.close()
file_aceleraciones.close()
file_enegia.close()

#Tiempo final.
t1 = time.time()

tiempo = t1 - t0

print("El tiempo de ejecución es: ", tiempo, "segundos")