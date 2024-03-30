import numpy as np
import matplotlib.pyplot as plt
import math


G = 6.674*10**(-11) 

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
data2 = open(r'C:\Users\jesol\OneDrive\Escritorio\Programacion\Programas\Compu2324\Obligatorio1\posiciones.txt', 'r').read().split('\n')

# Separar las velocidades en una lista
coordinates = [list(map(float, line.split(','))) for line in data if line]

# Convertir las posiciones en una lista de vectores de numpy
grouped_positions = [np.array([list(map(float, data2[i+j].split(','))) for j in range(n)]) for i in range(0, len(data2), n) if data2[i]]

# Separar las velocidades en dos listas
x, y = zip(*coordinates)


#Numero de filas
filas = len(x)

k = 0

# Calcular la energía cinética.
E_cin = np.zeros((n, int(filas/n)))

k = 0
i = 0
t = 0.0
while k < len(x):
    trun_t = t
    for i in range(n):
        if not t.is_integer():
            trun_t = math.trunc(t)
        E_cin[i][int(trun_t)] = 0.5 * m_rees[n] * ((x[k]**2 + y[k]**2))
        k += 1
        t += 1/n

#Calcular energía potencial.
E_pot = np.zeros((n, int(filas/n)))


k = 0
t = 0.0

for i in range(len(grouped_positions)):
    
    for j in range(n):
        for k in range(n):
            trun_t = t
            if not t.is_integer():
                trun_t = math.trunc(t)
            if j != k:
                E_pot[j][int(trun_t)] += (-m_rees[j]*m_rees[k])/(np.linalg.norm(grouped_positions[i][j]-grouped_positions[i][k])*2)*100
                E_pot[j][int(trun_t)] += (-m_rees[j])/(np.linalg.norm(grouped_positions[i][j])*2)*100
        t += 1/(n)     

# Initialize E_total array
E_total = np.zeros((n, int(filas/n)))

for i in range(n):
    for j in range(int(filas/n)):
        E_total[i][j] = E_cin[i][j] + E_pot[i][j]


# Graficar la energía cinética            
labels = ['Mercurio', 'Venus', 'Tierra', 'Marte', 'Jupiter', 'Saturno', 'Urano', 'Neptuno']

for i in range(n):
    plt.plot(E_cin[i], label='Energía cinética ' + labels[i])
    plt.plot(E_pot[i], label='Energía potencial ' + labels[i])
    plt.plot(E_total[i], label='Energía total ' + labels[i])    

# Añadir título y etiquetas
plt.xlabel('Index')
plt.ylabel('Energía')
plt.legend()

# Mostrar la gráfica
plt.show()

print(E_pot)