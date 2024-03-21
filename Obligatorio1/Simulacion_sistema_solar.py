#Simulación del sistema solar.
#Unidades del sistema internacional.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#Establecemos los uncrementos del tiempo.
h=0.01

#Constante de gravitación universal.
G = 6.674*10**(-11) #m^3/kg/s^2

#Radios de los planetas m.
r_mrcurio = np.array([4.6*10**11, 0])
r_venus = np.array([1.074*10**11, 0])
r_tierra = np.array([1.496*10**11, 0])
r_marte = np.array([2.279*10**11, 0])
r_jupiter = np.array([7.785*10**11, 0])      
r_saturno = np.array([1.429*10**12, 0])
r_urano = np.array([2.871*10**12, 0])
r_neptuno = np.array([4.495*10**12, 0])
r_pluton = np.array([5.906*10**12, 0])
radios = np.array([r_mrcurio, r_venus, r_tierra, r_marte, r_jupiter, r_saturno, r_urano, r_neptuno, r_pluton])

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
masas = np.array([m_mercurio, m_venus, m_tierra, m_marte, m_jupiter, m_saturno, m_urano, m_neptuno, m_pluton])

#Reescalamiento de las unidades de los datos.
r_rees = [radios[i][0]/r_tierra[0] for i in range(len(radios))]
t_rees = (G*m_sol/r_tierra[0]**3)**(1/2)
m_rees = [ m/m_sol for m in masas]

#Definimos resta de vectores.
def resta_vectorial(v1, v2):
    v3 = np.subtract(v1, v2)
    return v3

#Pedimos por consola el número de planetas con los que se ejcutará la simulación.
n = int(input("Ingrese el número de planetas con los que se ejecutará la simulación: "))

#Inicializamos la aceleración de cada planeta.
a = np.zeros((n, 2))
a_t = np.zeros((n, 2))


#Definimos la función a(t) a partir de la suma de fuerazas que se ejercen sobre cada partícula i.
def aceleracion_por_planeta(n, r_rees, m_rees, a, a_t):
    for i in range(n):
        for j in range(n):
            if i != j:
                a[i] = m_rees[j]*np.array(resta_vectorial(r_rees[i], r_rees[j]))/np.linalg.norm(resta_vectorial(r_rees[i], r_rees[j]))**3
                a_t[i] = a_t[i] + a[i]    
    return a_t

print(aceleracion_por_planeta(n, r_rees, m_rees, a, a_t))

a_i = aceleracion_por_planeta(n, r_rees, m_rees, a, a_t)


#Vector de velocidades iniciales de los planetas m/s.
v_mercurio = np.array([0, 4.7*10**4])
v_venus = np.array([0, 3.5*10**4])
v_tierra = np.array([0, 3*10**4])
v_marte = np.array([0, 2.4*10**4])
v_jupiter = np.array([0, 1.3*10**4])
v_saturno = np.array([0, 9.6*10**3])
v_urano = np.array([0, 6.8*10**3])
v_neptuno = np.array([0, 5.4*10**3])
v_pluton = np.array([0, 4.7*10**3])
velocidades = np.array([v_mercurio, v_venus, v_tierra, v_marte, v_jupiter, v_saturno, v_urano, v_neptuno, v_pluton])

#Reescalamiento de las unidades de los datos.
v_rees = [velocidades[i][1]/t_rees for i in range(len(velocidades))]



#Definimos la función w[i].
w_i = np.zeros((n, 2))

def w_i(n, velocidades, a_i, w_i, h):
    for i in range(n):
        w_i[i] = velocidades[i] + a_i[i]*h/2
    return w_i

#Definimos r(t+h) que nos da la nueva posición.
r_rees_th = np.zeros((n, 2))

def r_th(n, r_rees_th, velocidades, a_i, w_i, h):
    for i in range(n):
        r_rees_th[i] = r_rees_th[i] + w_i(velocidades[i], a_i[i], w_i[i], h)[i]*h
    return r_rees_th

a_i_th = np.zeros((n, 2))

#Definimos la función que nos da la acceleración en el tiempo t+h.
def a_i_th(n, r_rees_th, m_rees, a, a_i, h):
    for i in range(n):
        for j in range(n):
            if i != j:
                a[i] = m_rees[j]*np.array(resta_vectorial(r_th(r_rees_th[i], velocidades[i], a_i[i], w_i[i], h)[i], r_th(r_rees_th[i], velocidades[i], a_i[i], w_i[i], h)[j]))/np.linalg.norm(resta_vectorial(r_th(r_rees_th[i], velocidades[i], a_i[i], w_i[i], h)[i], r_th(r_rees_th[i], velocidades[i], a_i[i], w_i[i], h)[j]))**3
                a_i_th[i] = a_i_th[i] + a[i]    
    return a_i_th 

v_th = np.zeros((n, 2))

#Definimos la función que nos da la nueva velocidad en el tiempo t+h.
def v_th(velocidades, a_i_th, a_i, w_i, n, r_rees_th, m_rees, a, v_th, h):
    for i in range(n):
        v_th = w_i(velocidades[i], a_i[i], w_i[i], h)[i] + a_i_th(n, r_rees_th, m_rees, a, a_i, h)[i]*h/2
    return v_th

