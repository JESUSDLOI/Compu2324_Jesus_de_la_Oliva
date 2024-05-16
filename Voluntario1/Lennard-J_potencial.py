#Simulación de la dinámica molecular de un gas de partículas en 2D.
#Unidades del sistema internacional.

import numpy as np
import time
from numba import jit

#Establecemos el tiempo inicial.
t0 = time.time()

#Número de simulaciones.
simulaciones = 2

#Establecemos los uncrementos del tiempo.
h = 0.002

#Número de iteraciones.
iteraciones = 10000

#Número de iteraciones que se saltan para guardar los datos.
skip = 1

#Valor sigma
sigma = 3.4

#Pedimos el número de partículas.
n = 16

#Tamaño de caja
l = 10

#Interespaciado entre las partículas.
s = 2

#Variable para saber si las partículas se encuentran en un panal.
panal = False

#Reescalamiento de velocidades en tiempos específicos.
REESCALAMIENTO = False

#Temperatura crítica.
Temperatura_critica = False

if Temperatura_critica == True:
    REESCALAMIENTO = False

velocidad_en_x = True

#Disposición inicial de las partículas
@jit(nopython=True, fastmath=True)
def posiciones_iniciales(n, l, s, panal):
    posicion = np.zeros((n, 2)) + 1
    #Comprobamos si las partículas se encuentran en un panal.
    if panal == True:
        fila = 0
        paso = 0
        lado = (s**2 - (s/2)**2)**0.5
        for i in range(n-1):
            if paso % 2 == 0:
                paso += 1
                x = posicion[i][0] + 2*s
                if x > l:
                    fila += 1
                    if fila % 2 == 0:
                        paso = 0
                        posicion[i+1] = np.array([1, posicion[i][1] + lado])
                    else:
                        posicion[i+1] = np.array([(1+s/2), posicion[i][1] + lado])
                else:
                    posicion[i+1] = np.array([posicion[i][0] + 2*s, posicion[i][1]])
                
            else:
                paso += 1
                x = posicion[i][0] + s
                if x > l:
                    fila += 1
                    if fila % 2 == 0:
                        paso = 0
                        posicion[i+1] = np.array([1, posicion[i][1] + lado])
                    else:
                        posicion[i+1] = np.array([(1+s/2), posicion[i][1] + lado])
                else:
                    posicion[i+1] = np.array([posicion[i][0] + s, posicion[i][1]])
                    
    #Si las partículas no se encuentran en un panal, se disponen en una cuadrícula.               
    else:   
        for i in range(n-1):
            x = posicion[i][0] + s
            if x > l:
                posicion[i+1] = np.array([1, posicion[i][1] + s])
            else:
                posicion[i+1] = np.array([posicion[i][0] + s, posicion[i][1]])
    return posicion

#Condiciónes de contorno periódicas. Y calculamos el momento transferido a la caja.
@jit(nopython=True, fastmath=True)
def contorno(posiciones, l, n, velocidades):
    momento = np.zeros((n, 2))
    for i in range(n):
        x = posiciones[i][0]
        y = posiciones[i][1]
        if x > l:
            x = x % l
            momento[i][0] = velocidades[i][0]
        if x < 0:
            x = x % l
            momento[i][0] = velocidades[i][0]
        if y > l:
            y = y % l
            momento[i][1] = velocidades[i][1]
        if y < 0:
            y = y % l
            momento[i][1] = velocidades[i][1]
        posiciones[i][0] = x
        posiciones[i][1] = y
    return posiciones, 2*momento

#Función para calcular la distancia entre dos partículas.
@jit(nopython=True, fastmath=True)
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
@jit(nopython=True, fastmath=True)
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

#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
@jit(nopython=True, fastmath=True)
def velocidad_th(w_i, n, velocidades, a_i_th, h):
    E_c = 0
    for i in range(n):
        velocidades[i] = w_i[i] + a_i_th[i]*(h/2)
        E_c += energia_cinetica(velocidades[i])
    return velocidades, E_c

#Definimos la energía cinética.
@jit(nopython=True, fastmath=True)
def energia_cinetica(velocidades):
    energia_cinetica = (0.5)*(velocidades[0]**2 + velocidades[1]**2)
    return energia_cinetica

#Definimos la energía cinética de cada partícula.
@jit(nopython=True, fastmath=True)
def energia_cinetica_inicial(velocidades, n):
    E_c = 0
    for i in range(n):
        E_c += energia_cinetica(velocidades[i])
    return E_c

#Definimos la función que escribirá los datos en los archivos.
def guardar_datos(k, posiciones, energia, skip, E_c, E_p, velocidades, presion):
    if k % skip == 0:
        np.savetxt(file_posiciones, posiciones, delimiter=",")
        file_posiciones.write("\n")
        file_energia_cinetica.write(str(E_c) + "\n")
        file_energia_potencial.write(str(E_p) + "\n")
        file_energia.write(str(energia) + "\n")
        np.savetxt(file_velocidades, velocidades, delimiter=",")
        file_presion.write(str(presion) + "\n")

def reescalamiento(velocidades, k, h, posiciones, posicion_inicial, fluctuacion_total, q, fluctuacion):
    fluctuacion = np.linalg.norm(posiciones[0] - posicion_inicial[0])**2
    file_fluctuacion.write(str(np.sum(fluctuacion)) + "\n")
    fluctuacion_total += fluctuacion
    q += 1
    if k*h in [20, 30, 35, 45]:
        velocidades = velocidades * 1.5
        fluctuacion_total = fluctuacion_total / q
        print("Fluctuación total: ", fluctuacion_total)
        file_fluctuacion.write(str("\n"))
        fluctuacion_total = 0   
        q = 0


# Realizamos el bucle para calcular los datos de la simulación.
def simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c, skip, momento, REESCALAMIENTO):
    E_c_total = 0
    E_p_total = 0
    T = 0
    presion_media = 0
    presion1 = np.zeros((n, 2))
    presion2 = np.zeros((n, 2))
    presion = 0
    posicion_inicial = posiciones[0].copy()
    fluctuacion_total = 0
    q = 0
    fluctuacion = 0
    for k in range(iteraciones):

        guardar_datos(k, posiciones, energia, skip, E_c, E_p, velocidades, presion)
        
        #Calculamos la presión antes de actualizar las posiciones.
        presion1 = momento
        
        #Realizamos el algoritmo de Verlet.
        w_i = w_ih(n, velocidades, a_i, w_i, h)
        posiciones = p_th(n, posiciones, w_i, h)
        posiciones, momento = contorno(posiciones, l, n, velocidades)
        a_i, E_p = acel_i_th(n, posiciones, a_i, l, E_p_c, a_c)
        velocidades, E_c = velocidad_th(w_i, n, velocidades, a_i, h)
        
        #Calculamos la presión después de actualizar las posiciones.
        presion2 = momento

        #Calculamos la energía cinética y potencial total de la suimulacion
        E_c_total += E_c 
        E_p_total += E_p
        
        #Calculamos la energía en cada instante de tiempo.
        energia = E_c + E_p

        #Calculamos la fuerza.
        fuerza = (presion2 - presion1) / (h)
        
        #Calculamos la presión.
        presion = np.sum(abs(fuerza)) / (4*l)
        presion_media += presion
        
        #Reescalamos las velocidades en tiempos específicos.
        
        if REESCALAMIENTO == True:
            reescalamiento(velocidades, k, h, posiciones, posicion_inicial, fluctuacion_total, q, fluctuacion)
   
        
        #Calculamos la temperatura crítica.
        if Temperatura_critica == True:
            separacion, direcc = distancia_condiciones(posiciones, 0, 1, l)
            
            
    #Calculamos la energía cinética y potencial promedio de la simulación.
    E_c_total = E_c_total/(iteraciones)
    E_p_total = E_p_total/(iteraciones)
    print("Energía cinética promedio: ", E_c_total)
    print("Energía potencial promedio: ", E_p_total)
    #Calculamos la temperatura promedio de la simulación.
    T = E_c_total 
    print("Temperatura promedio: ", T)
    #Calculamos la presión promedio de la simulación.
    presion_media = presion_media/(iteraciones)
    print("Presión promedio: ", presion_media)
    file_presion_temp.write(str(presion_media) + "," + str(T) + "\n")
    
    
    
file_presion_temp = open('presion_temp.dat', "w")  
    
#Bucle para realizar las simulaciones.
for z in range(simulaciones):   
    
    # Abrir tres archivos para guardar los datos de las posiciones, velocidades y energía.
    file_posiciones = open('posiciones_part' + str(z) + '.dat', "w")
    file_energia = open('energia_part' + str(z) + '.dat', "w")
    file_energia_cinetica = open('energia_cinetica' + str(z) + '.dat', "w")
    file_energia_potencial = open('energia_potencial' + str(z) + '.dat', "w")
    file_velocidades = open('velocidades_part' + str(z) + '.dat', "w")
    file_presion = open('presion' + str(z) + '.dat', "w")
    if REESCALAMIENTO == True:
        file_fluctuacion = open('fluctuacion_temperatura' + str(z) + '.dat', "w")
    if Temperatura_critica == True:
        file_temperatura_critica = open('temperatura_critica' + str(z) + '.dat', "w")

    
    #Inicializamos las variables que se utilizarán en el bucle.
    #Definimos las velocidades iniciales de las partículas.
    velocidades = np.zeros((n, 2))
    for i in range(n):
        velocidades[i][0] = (2*np.random.rand() - 1) 
        velocidades[i][1] = np.random.choice([-1, 1])*np.sqrt(1 - velocidades[i][0]**2) 

    #Aumentamos la velocidad de las partículas según la simulación
    velocidades = np.array(velocidades) * z
    if velocidad_en_x == True:
        velocidades[:, 0] = 1
        velocidades[:, 1] = 0
        
    posiciones = posiciones_iniciales(n, l, s, panal)
    posiciones, momento = contorno(posiciones, l, n, velocidades)
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
    #Ejecutamos la simulación.
    simulacion(n, posiciones, velocidades, a_i, w_i, h, iteraciones, l, E_p, E_c, energia, E_p_c, a_c, skip, momento, REESCALAMIENTO)
    # Cerrar los archivos
    file_posiciones.close()
    file_energia.close()
    file_energia_cinetica.close()
    file_energia_potencial.close()
    file_velocidades.close()
    file_presion.close()
    if REESCALAMIENTO == True:
        file_fluctuacion.close()
    if Temperatura_critica == True:
        file_temperatura_critica.close()
    
file_presion_temp.close()


#Tiempo final.
t1 = time.time()

tiempo = t1 - t0

print("El tiempo de ejecución es: ", tiempo, "segundos")