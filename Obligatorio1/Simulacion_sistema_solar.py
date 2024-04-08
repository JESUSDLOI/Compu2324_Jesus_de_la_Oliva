#Simulación del sistema solar.
#Unidades del sistema internacional.

import numpy as np
import time
from numba import jit

#Establecemos el tiempo inicial.

t0 = time.time()

#Establecemos los uncrementos del tiempo.
h=0.0001

#Número de iteraciones.
iteraciones = 10000

#Pedimos el número de planetas con los que se ejcutará la simulación.
n = 6

#Constante de gravitación universal.
G = 6.674*10**(-11) 

#Distancia al Sol de los planetas m.
r_sol = np.array([0, 0])
r_mrcurio = np.array([5.791*10**10, 0])
r_venus = np.array([1.082*10**11, 0])
r_tierra = np.array([1.496*10**11, 0])
r_marte = np.array([2.279*10**11, 0])
r_jupiter = np.array([7.786*10**11, 0])      
r_saturno = np.array([1.433*10**12, 0])
r_urano = np.array([2.871*10**12, 0])
r_neptuno = np.array([4.495*10**12, 0])
r_pluton = np.array([5.906*10**12, 0])
radios = np.array([r_sol, r_mrcurio, r_venus, r_tierra, r_marte, r_jupiter, r_saturno, r_urano, r_neptuno, r_pluton])

#Masas de los planetas Kg
m_sol = 1.989*10**30
m_mercurio = 3.3*10**23
m_venus = 4.87*10**24
m_tierra = 5.97*10**24
m_marte = 6.42*10**23
m_jupiter = 1.898*10**27
m_saturno = 5.68*10**26
m_urano = 8.68*10**25
m_neptuno = 1.02*10**26
m_pluton = 1.3*10**22
masas = np.array([m_sol, m_mercurio, m_venus, m_tierra, m_marte, m_jupiter, m_saturno, m_urano, m_neptuno, m_pluton])

#Vector de velocidades iniciales de los planetas m/s.
v_sol = np.array([0, 0])
v_mercurio = np.array([0, -4.7*10**4])
v_venus = np.array([0, -3.5*10**4])
v_tierra = np.array([0, 3*10**4])
v_marte = np.array([0, 2.4*10**4])
v_jupiter = np.array([0, 1.3*10**4])
v_saturno = np.array([0, 9.6*10**3])
v_urano = np.array([0, 6.8*10**3])
v_neptuno = np.array([0, 5.4*10**3])
v_pluton = np.array([0, 4.7*10**3])
velocidades = np.array([v_mercurio, v_venus, v_tierra, v_marte, v_jupiter, v_saturno, v_urano, v_neptuno, v_pluton])

#Reescalamiento de las unidades de los datos.
reescalado_v = [velocidades[i][1]*np.sqrt(r_tierra[0]/(G*m_sol)/2) for i in range(n)]
v_rees = np.array([[0, v] for v in reescalado_v])   

#Posiciones reescaladas.
reescalado_r = [radios[i][0]/r_tierra[0] for i in range(n)]
r_rees = np.array([[r, 0] for r in reescalado_r])

#Reescalamos las masas.
m_rees = [ m/m_sol for m in masas]

#Definimos la función a(t) a partir de la suma de fuerazas que se ejercen sobre cada partícula i.
@jit
def aceleracion_por_planeta(n, r_rees, m_rees, a_t):
    for i in range(n):
        for j in range(n):
            if i != j:
                a_t[i] += m_rees[j]*np.array(r_rees[i] - r_rees[j])/np.linalg.norm(r_rees[i] - r_rees[j])**3  
    return -a_t

#Definimos la función w[i].
@jit
def w_ih(n, v_rees, a_i, w_i, h):
    for i in range(n):
        w_i[i] = v_rees[i] + a_i[i]*(h/2)
    return w_i

#Definimos r(t+h) que nos da la nueva posición.
@jit
def r_th(n, r_rees_th, w_i, h):
    for i in range(n):
        r_rees_th[i] = r_rees_th[i] + w_i[i]*h
    return r_rees_th

#Definimos la función que nos da la acceleración en el tiempo t+h.
@jit
def acel_i_th(n, r_rees_th, m_rees, a_i_th):
    for i in range(n):
        for j in range(n):
            if i != j:
                 a_i_th[i] += m_rees[j]*np.array(r_rees[i] - r_rees[j])/np.linalg.norm(r_rees[i] - r_rees[j])**3    
    return -a_i_th 


#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
@jit
def velocidad_th(w_i, n, v_th, a_i_th, h):
    for i in range(n):
        v_th[i]= w_i[i] + a_i_th[i]*(h/2)
    return v_th


#Inicializamos las variables que se utilizarán en el bucle.
a_t = np.zeros((n, 2))
a_i = aceleracion_por_planeta(n, r_rees, m_rees, a_t)
r_rees_th = r_rees
w_i = np.zeros((n, 2))
a_i_th = np.zeros((n, 2))
v_th = v_rees

# Abrir tres archivos para guardar los datos de las posiciones, velocidades y aceleraciones
file_posiciones = open('posiciones.dat', "w")
file_velocidades = open('velocidades.dat', "w")
file_aceleraciones = open('aceleraciones.dat', "w")

# Realizamos el bucle para calcular las posiciones y velocidades de los planetas.
for k in range(iteraciones):

    if k == 0 or k % (n*100) == 0:
        np.savetxt(file_posiciones, r_rees, delimiter=",")
        np.savetxt(file_velocidades, v_rees, delimiter=",")
        np.savetxt(file_aceleraciones, a_i, delimiter=",")
        file_posiciones.write("\n")
        file_velocidades.write("\n")
        file_aceleraciones.write("\n")

    w_i = w_ih(n, v_rees, a_i, w_i, h)
    r_rees_th = r_th(n, r_rees, w_i, h)
    a_i_th = acel_i_th(n, r_rees_th, m_rees, a_i)
    v_th = velocidad_th(w_i, n, v_rees, a_i_th, h)
    r_rees = r_rees_th
    v_rees = v_th
    a_i = a_i_th

# Cerrar los archivos
file_posiciones.close()
file_velocidades.close()
file_aceleraciones.close()

#Tiempo final.
t1 = time.time()

tiempo = t1 - t0

print("El tiempo de ejecución es: ", tiempo, "segundos")