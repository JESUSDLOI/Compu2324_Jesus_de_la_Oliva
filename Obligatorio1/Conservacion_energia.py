import numpy as np
import matplotlib.pyplot as plt

#Masas de los planetas
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
m_rees = [ m/m_sol for m in masas]

#Numero de planetas
n = 4

# Leer las velocidades de un archivo de texto
data = open(r'C:\Users\jesol\OneDrive\Escritorio\Programacion\Programas\Compu2324\Obligatorio1\velocidades.txt', 'r').read().split('\n')

# Separar las velocidades en una lista
coordinates = [list(map(float, line.split(','))) for line in data if line]

# Separar las velocidades en dos listas
x, y = zip(*coordinates)

#Numero de filas
filas = len(x)

k = 0

# Calcular la energía cinética 
E_cin = [[0] * len(x) for _ in range(n)]

k = 0
i = 0
t = 0
while k < len(x):
    if i == n:
        i = 0
    E_cin[i][t] = 0.5 * m_rees[n] * (x[k]**2 + y[k]**2)
    k += 1
    i += 1
    t += 1
        

        

# Graficar la energía cinética
for i in range(n):
    plt.plot(E_cin[i], label=f'Planet {i+1}')

# Añadir título y etiquetas
plt.xlabel('Index')
plt.ylabel('E_cin')
plt.legend()

# Mostrar la gráfica
plt.show()
