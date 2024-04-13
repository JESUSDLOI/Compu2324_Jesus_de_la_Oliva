#Simulación del sistema solar.
#Unidades del sistema internacional.

import numpy as np
import time
from numba import jit

#Establecemos los uncrementos del tiempo.
h = 0.0001

# Número de iteraciones.
iteraciones = 100000

# Pedimos el número de planetas con los que se ejecutará la simulación.
n = int(9)

# Constante de gravitación universal.
G = 6.674*10**(-11) 

# Distancia al Sol de los planetas m.
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
m_tierra = 5.98*10**24
m_marte = 6.42*10**23
m_jupiter = 1.898*10**27
m_saturno = 5.68*10**26
m_urano = 8.68*10**25
m_neptuno = 1.02*10**26
m_pluton = 1.3*10**22
masas = np.array([m_sol, m_mercurio, m_venus, m_tierra, m_marte, m_jupiter, m_saturno, m_urano, m_neptuno, m_pluton])

#Vector de velocidades iniciales de los planetas m/s.
v_sol = np.array([0, 0])
v_mercurio = np.array([0, 4.7*10**4])
v_venus = np.array([0, 3.5*10**4])
v_tierra = np.array([0, 3*10**4])
v_marte = np.array([0, 2.4*10**4])
v_jupiter = np.array([0, 1.3*10**4])
v_saturno = np.array([0, 9.6*10**3])
v_urano = np.array([0, 6.8*10**3])
v_neptuno = np.array([0, 5.4*10**3])
v_pluton = np.array([0, 4.7*10**3])
velocidades = np.array([v_sol, v_mercurio, v_venus, v_tierra, v_marte, v_jupiter, v_saturno, v_urano, v_neptuno, v_pluton])

@jit(nopython=True)
def v_esc(velocidades, G, r_tierra, m_sol, n):
    #Reescalamiento de las unidades de los datos.
    v_rees = np.zeros((n, 2))
    for i in range(n):
        v_rees[i][1] = velocidades[i][1]*np.sqrt(r_tierra[0]/(G*m_sol)/1.98)
    return v_rees

@jit(nopython=True)
def r_esc(radios, r_tierra, n):
    #Posiciones reescaladas.
    r_rees = np.zeros((n, 2))
    for i in range(n):
        r_rees[i][0] = radios[i][0]/r_tierra[0]
    return r_rees
        
@jit(nopython=True)        
def m_esc(masas, m_sol, n):
    #Reescalamos las masas.
    m_rees = np.zeros(n)
    for i in range(n):
        m_rees[i] = masas[i]/m_sol
    return m_rees


#Definimos la función a(t) a partir de la suma de fuerazas que se ejercen sobre cada partícula i.
@jit(nopython=True)
def aceleracion_por_planeta(n, r_rees, m_rees, a_t):
    for i in range(n):
        for j in range(n):
            if i != j:
                distancia = (r_rees[i] - r_rees[j])
                norma = np.linalg.norm(distancia)
                a_t[i] += m_rees[j]*distancia/(norma)**3  
    return -a_t

#Definimos la función w[i].
@jit(nopython=True)
def w_ih(n, v_rees, a_i, w_i, h):
    for i in range(n):
        w_i[i] = v_rees[i] + a_i[i]*(h/2)
    return w_i

#Definimos r(t+h) que nos da la nueva posición.
@jit(nopython=True)
def r_th(n, r_rees_th, w_i, h):
    for i in range(n):
        r_rees_th[i] = r_rees_th[i] + w_i[i]*h
    return r_rees_th

#Definimos la función que nos da la acceleración en el tiempo t+h.
@jit(nopython=True)
def acel_i_th(n, r_rees_th, m_rees, a_i_th):
    for i in range(n):
        for j in range(n):
            if i != j:
                 distancia = (r_rees_th[i] - r_rees_th[j])
                 norma = np.linalg.norm(distancia)
                 a_i_th[i] += m_rees[j]*distancia/(norma)**3
    return -a_i_th 


#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
@jit(nopython=True)
def velocidad_th(w_i, n, v_th, a_i_th, h):
    for i in range(n):
        v_th[i]= w_i[i] + a_i_th[i]*(h/2)
    return v_th

#Calcular periodo de las órbitas.
@jit(nopython=True)
def periodo_orbital(r_rees, periodo, k):
        for i in range(n):     
            if r_rees[i][1] < 0 and periodo[i] == 0:
                periodo[i] += k*2


#Inicializamos las variables que se utilizarán en el bucle.
@jit(nopython=True)
def inicializar_variables(n, velocidades, G, r_tierra, m_sol, radios, masas):
    v_rees = v_esc(velocidades, G, r_tierra, m_sol, n)
    r_rees = r_esc(radios, r_tierra, n)
    m_rees = m_esc(masas, m_sol, n)
    a_t = np.zeros((n, 2))
    a_i = aceleracion_por_planeta(n, r_rees, m_rees, a_t)
    w_i = np.zeros((n, 2))
    a_i_th = np.zeros((n, 2))
    
    return v_rees, r_rees, m_rees, a_i, w_i, a_i_th



def guardar_datos(k, n, r_rees, v_rees):
    if k % (n*100) == 0:
        np.savetxt(file_posiciones, r_rees, delimiter=",")
        np.savetxt(file_velocidades, v_rees, delimiter=",")
        file_posiciones.write("\n")
        file_velocidades.write("\n")

k = 0
# Realizamos el bucle para calcular las posiciones y velocidades de los planetas.
def simulacion(n, r_rees, v_rees, a_i, w_i, h, iteraciones, m_rees, k):
    
    for k in range(iteraciones):

        
        guardar_datos(k, n, r_rees, v_rees)

        w_i = w_ih(n, v_rees, a_i, w_i, h)
        r_rees_th = r_th(n, r_rees, w_i, h)
        a_i_th = acel_i_th(n, r_rees_th, m_rees, a_i)
        v_th = velocidad_th(w_i, n, v_rees, a_i_th, h)
        r_rees = r_rees_th
        v_rees = v_th
        a_i = a_i_th
        
d_n_p =  r_pluton - r_neptuno        
tiempo = 0
file = open('Tiempo_planetas.dat', 'w')
for u in range(50):
    #Establecemos el tiempo inicial.

    t0 = time.time()

    #Inicializamos las variables.
    v_rees, r_rees, m_rees, a_i, w_i, a_i_th = inicializar_variables(n, velocidades, G, r_tierra, m_sol, radios, masas)

    # Abrir tres archivos para guardar los datos de las posiciones, velocidades y aceleraciones
    file_posiciones = open('posiciones.dat', "w")
    file_velocidades = open('velocidades.dat', "w")

    k = 0
    #Ejecutamos la simulación.
    simulacion(n, r_rees, v_rees, a_i, w_i, h, iteraciones, m_rees, k)
        
    #Tiempo final.
    t1 = time.time()

    tiempo = t1 - t0
    
    #Guardamos los datos en un archivo de texto
    np.savetxt(file, np.column_stack((n, tiempo)), delimiter=",")
    print("El tiempo de ejecución para", n ,"planetas es: ", tiempo, "segundos")
    
    #Añadimos un planeta.
    nueva_velocidad = velocidades[n]-np.array([0, 0.7*10**3])
    distancia_nuevo_planeta = radios[n] + d_n_p
    n += 1
    masas = np.append(masas, m_pluton)
    radios = np.vstack((radios, distancia_nuevo_planeta))
    velocidades = np.vstack((velocidades, nueva_velocidad))

    file_posiciones.close()
    file_velocidades.close()
    

file.close()