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
r_mrcurio = (4.6*10**11, 0)
r_venus = (1.074*10**11, 0)
r_tierra = (1.496*10**11, 0)
r_marte = (2.279*10**11, 0)
r_jupiter = (7.785*10**11, 0)      
r_saturno = (1.429*10**12, 0)
r_urano = (2.871*10**12, 0)
r_neptuno = (4.495*10**12, 0)
r_pluton = (5.906*10**12, 0)
radios = [r_mrcurio, r_venus, r_tierra, r_marte, r_jupiter, r_saturno, r_urano, r_neptuno, r_pluton]

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
masas = [m_sol, m_mercurio, m_venus, m_tierra, m_marte, m_jupiter, m_saturno, m_urano, m_neptuno, m_pluton]

#Reescalamiento de las unidades de los datos.
r_rees = [radios/r_tierra for i in radios]
t_rees = (G*m_sol/r_tierra**3)**(1/2)
m_rees = [masas/m_sol for m in masas]

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
for i in range(n):
    for j in range(n):
        if i != j:
            a[i] = m_rees[j]*np.array(resta_vectorial(r_rees[i], r_rees[j]))/np.linalg.norm(resta_vectorial(r_rees[i], r_rees[j]))**3
            a_t[i] = a_t[i] + a[i]    

print(a_t)



