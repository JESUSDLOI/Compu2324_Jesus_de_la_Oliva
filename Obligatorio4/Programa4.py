#El problema de los tres cuerpos: Algoritmo de Runge-Kutta

import numpy as np
import time
from numba import jit

t0 = time.time()

# Definir las constantes del problema
MT = 5.9736*10**24                  # Masa de la Tierra
R_T = 6.37816*10**6                 # Radio de la Tierra
ML = 0.07349*10**24                 # Masa de la Luna
R_L = 1.7374*10**6                  # Radio de la Luna
m = 47000                           # Masa de la nave espacial
G = 6.67*10**(-11)                  # Constante gravitacional
dT_L = 3.844*10**8                  # Distancia entre la Tierra y la Luna
omega = 2.6617*10**(-6)             # Velocidad angular de la Luna
delta = G*MT/(dT_L**3)              # Parámetro adimensional
mu = ML/MT                          # Parámetro adimensional
t = 0                               # Tiempo transcurrido de la simulación
T = 300000                          # Tiempo total de la simulación
h = 0.1                             # Paso de tiempo
k = np.zeros((4, 4))                # Matriz de coeficientes de Runge-Kutta

# Definir las variables iniciales de la nave espacial
v = np.sqrt(2*G*MT/R_T) / dT_L      # Velocidad inicial de la nave espacial
theta = np.pi / 11.627               # Ángulo despege inicial de la nave espacial
r = R_T/dT_L                        # Distancia inicial de la nave espacial
phi = 0                             # Ángulo inicial con respecto a la tierra de la nave espacial
p_r = v * np.cos(theta - phi)       # Momento inicial de la nave espacial
p_phi = r * v * np.sin(theta - phi) # Momento inicial de la nave espacial
y = np.array([r, phi, p_r, p_phi])  # Vector de posiciones y momentos iniciales de la nave espacial
radio_luna = R_L / dT_L                    # Radio de la Luna

# Distancia de la nave espacial a la Luna
@jit(nopython=True, fastmath=True)
def calc_r_primado(r, phi, omega, t):
    return (1 + r**2 - 2 * r * np.cos(phi - omega*t))**(0.5)

# Definir las funciones de las ecuaciones diferenciales
@jit(nopython=True, fastmath=True)
def calculo_r_punto(p_r):
    return p_r

@jit(nopython=True, fastmath=True)
def calculo_phi_punto(p_phi, r):
    return p_phi / r**2

@jit(nopython=True, fastmath=True)
def calculo_p_r_punto(r, r_primado, phi, p_phi, delta, mu, omega, t):
    return p_phi**2/r**3 - delta*((1/r**2) + (mu/r_primado**3)*(r - np.cos(phi - omega*t)))

@jit(nopython=True, fastmath=True)
def calculo_p_phi_punto(r, r_primado, phi, delta, mu, omega, t):
    return -(delta * mu * r)/(r_primado**3) * np.sin(phi - omega*t)


@jit(nopython=True, fastmath=True)
def paso_runge_kutta(t, y, k, r_primado, delta, mu, omega, h):
    
    k[0][0] = h * calculo_r_punto(y[2])
    k[0][1] = h * calculo_phi_punto(y[3], y[0])
    k[0][2] = h * calculo_p_r_punto(y[0], r_primado, y[1], y[3], delta, mu, omega, t)
    k[0][3] = h * calculo_p_phi_punto(y[0], r_primado, y[1], delta, mu, omega, t)
    
    k[1][0] = h * calculo_r_punto(y[2] + k[0][2]/2)
    k[1][1] = h * calculo_phi_punto(y[3] + k[0][3]/2, y[0] + k[0][0]/2)
    k[1][2] = h * calculo_p_r_punto(y[0] + k[0][0]/2, r_primado, y[1] + k[0][1]/2, y[3] + k[0][3]/2, delta, mu, omega, t + h/2)
    k[1][3] = h * calculo_p_phi_punto(y[0] + k[0][0]/2, r_primado, y[1] + k[0][1]/2, delta, mu, omega, t + h/2)
    
    k[2][0] = h * calculo_r_punto(y[2] + k[1][2]/2)
    k[2][1] = h * calculo_phi_punto(y[3] + k[1][3]/2, y[0] + k[1][0]/2)
    k[2][2] = h * calculo_p_r_punto(y[0] + k[1][0]/2, r_primado, y[1] + k[1][1]/2, y[3] + k[1][3]/2, delta, mu, omega, t + h/2)
    k[2][3] = h * calculo_p_phi_punto(y[0] + k[1][0]/2, r_primado, y[1] + k[1][1]/2, delta, mu, omega, t + h/2)
    
    k[3][0] = h * calculo_r_punto(y[2] + k[2][2])
    k[3][1] = h * calculo_phi_punto(y[3] + k[2][3], y[0] + k[2][0])
    k[3][2] = h * calculo_p_r_punto(y[0] + k[2][0], r_primado, y[1] + k[2][1], y[3] + k[2][3], delta, mu, omega, t + h)
    k[3][3] = h * calculo_p_phi_punto(y[0] + k[2][0], r_primado, y[1] + k[2][1], delta, mu, omega, t + h)
    
    return y + (k[0] + 2*k[1] + 2*k[2] + k[3])/6    
    
file_posicion_sistema = open("posicion_nave.dat", "w")
file_H = open("H_h_fija.dat", "w")
file_H_ajustada = open("H_h_ajustada.dat", "w")

def escribir_posicion(posicion_nave, posicion_luna, file_posicion_sistema):
    np.savetxt(file_posicion_sistema, [posicion_nave], delimiter="," )
    np.savetxt(file_posicion_sistema, [posicion_luna], delimiter="," )
    file_posicion_sistema.write("\n")


def escribir_H_fija(H, file_H):
    file_H.write(str(H) + "\n")
    
def escribir_H_ajustada(H, file_H_ajustada):
    file_H_ajustada.write(str(H) + "\n")

i = 0

# Realizar la simulación
def simulacion_h_fija(T, h, y, k, delta, mu, omega, i, t, file_posicion_sistema, m, G, MT, ML, dT_L, file_H, radio_luna):
    while t < T:

        posicion_nave = np.array([y[0]*np.cos(y[1]), y[0]*np.sin(y[1])]) # Posición de la nave en x, y
        posicion_luna = np.array([np.cos(omega*t), np.sin(omega*t)]) # Posición de la Luna en x, y

        r_primado = calc_r_primado(y[0], y[1], omega, t) # Distancia de la nave espacial a la Luna reescalada.

        r_L = ((y[0]*dT_L)**2 + dT_L**2 - 2*y[0]*dT_L**2*np.cos(y[1]-omega*t))**(1/2) # Distancia de la nave espacial a la Luna sin reescalar
        
        if r_L < radio_luna:
            print("La nave ha chocado con la Luna")
            break

        H = (y[2]*m*dT_L)**2/(2*m) + (y[3]*m*dT_L**2)**2/(2*m*(y[0]*dT_L)**2) - G*MT*m/(y[0]*dT_L) - G*ML*m/r_L  - omega * (y[3]*m*dT_L**2) # Hamiltoniano

        if i % 500 == 0:
            escribir_posicion(posicion_nave, posicion_luna, file_posicion_sistema)
            escribir_H_fija(H, file_H)

        y = paso_runge_kutta(t, y, k, r_primado, delta, mu, omega, h)   # Calcular el siguiente paso de Runge-Kutta

        t += h      # Incrementar el tiempo
        i += 1      # Incrementar el contador de iteraciones
    file_posicion_sistema.close()
    
    
def autoajuste_h(h, y, k, delta, mu, omega, t, r_primado, y_medio, k_medio):
    
    e = 0.1**5
    y_medio = paso_runge_kutta(t, y_medio, k_medio, r_primado, delta, mu, omega, h/2)
    y = paso_runge_kutta(t, y, k, r_primado, delta, mu, omega, h)   # Calcular el siguiente paso de Runge-Kutta
    
    epsilon =  np.max(16 * np.abs(y_medio - y) / 15)  # Calcular el error

    s = max((epsilon/(e))**(0.2), 10**(-8)) # Calcular el factor de ajuste
    h_max = h / s # Calcular el nuevo paso de tiempo
    
    return h_max, y_medio, s, y

def pasos_autoajuste_h(h, y, k, delta, mu, omega, t, r_primado, y_medio, k_medio, posicion_luna):
    
    h_max, y_medio, s, y = autoajuste_h(h, y, k, delta, mu, omega, t, r_primado, y_medio, k_medio)
    
    while s > 2:
        h = h_max
        h_max, y_medio, s, y = autoajuste_h(h, y, k, delta, mu, omega, t, r_primado, y_medio, k_medio)
        
    if s < 2:
        y = y_medio
        posicion_luna = np.array([np.cos(omega*t), np.sin(omega*t)]) # Posición de la Luna en x, y
        t += h  # Incrementar el tiempo
    if h < h_max:
        h = 2 * h
            
    return h, y, t, y_medio, posicion_luna

def simulacion_h_ajustada(T, h, y, k, k_medio, delta, mu, omega, i, t, file_posicion_sistema, m, G, MT, ML, dT_L, y_medio, file_H_ajustada):
    i = 0
    posicion_luna = np.array([np.cos(omega*t), np.sin(omega*t)]) # Posición de la Luna en x, y
    while t < T:

        posicion_nave = np.array([y[0]*np.cos(y[1]), y[0]*np.sin(y[1])]) # Posición de la nave en x, y
                
        r_primado = calc_r_primado(y[0], y[1], omega, t) # Distancia de la nave espacial a la Luna
        
        r_L = ((y[0]*dT_L)**2 + dT_L**2 - 2*y[0]*dT_L**2*np.cos(y[1]-omega*t))**(1/2) # Distancia de la nave espacial a la Luna

        H = (y[2]*m*dT_L)**2/(2*m) + (y[3]*m*dT_L**2)**2/(2*m*(y[0]*dT_L)**2) - G*MT*m/(y[0]*dT_L) - G*ML*m/r_L  - omega * (y[3]*m*dT_L**2) # Hamiltoniano
        
        if i % 500 == 0:    
            escribir_posicion(posicion_nave, posicion_luna, file_posicion_sistema)
            escribir_H_ajustada(H, file_H_ajustada)

        i += 1      # Incrementar el contador de iteraciones
        h, y, t, y_medio, posicion_luna = pasos_autoajuste_h(h, y, k, delta, mu, omega, t, r_primado, y_medio, k_medio, posicion_luna)   # Calcular el siguiente paso de Runge-Kutta

# Simulación con paso de tiempo fijo

simulacion_h_fija(T, h, y, k, delta, mu, omega, i, t, file_posicion_sistema, m, G, MT, ML, dT_L, file_H, radio_luna)


# Simulación con paso de tiempo ajustado

file_posicion_sistema = open("posicion_nave_h_ajus.dat", "w")
i = 0
k = np.zeros((4, 4))                # Matriz de coeficientes de Runge-Kutta
k_medio = k.copy()                # Matriz de coeficientes de Runge-Kutta para el paso medio
t = 0                               # Tiempo transcurrido de la simulación

# Definir las variables iniciales de la nave espacial
v = np.sqrt(2*G*MT/R_T) / dT_L      # Velocidad inicial de la nave espacial
theta = np.pi / 11.627             # Ángulo despege inicial de la nave espacial
r = R_T/dT_L                        # Distancia inicial de la nave espacial
phi = 0                             # Ángulo inicial con respecto a la tierra de la nave espacial
p_r = v * np.cos(theta - phi)       # Momento inicial de la nave espacial
p_phi = r * v * np.sin(theta - phi) # Momento inicial de la nave espacial
y = np.array([r, phi, p_r, p_phi])  # Vector de posiciones y momentos iniciales de la nave espacial
y_medio = y.copy()
h = 0.1                              # Paso de tiempo

simulacion_h_ajustada(T, h, y, k, k_medio, delta, mu, omega, i, t, file_posicion_sistema, m, G, MT, ML, dT_L, y_medio, file_H_ajustada)
file_posicion_sistema.close()

t1 = time.time()

print("Tiempo de ejecución: ", t1 - t0)