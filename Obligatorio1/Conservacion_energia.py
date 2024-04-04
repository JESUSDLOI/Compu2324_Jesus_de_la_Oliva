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
n = 8

# Leer las velocidades de un archivo de texto
with open(r'C:\Users\jesol\OneDrive\Escritorio\Programacion\Programas\Compu2324\Obligatorio1\velocidades.dat', 'r') as f:
    v = [[list(map(float, line.split(','))) for line in block.split('\n') if line] for block in f.read().split('\n\n')]

with open(r'C:\Users\jesol\OneDrive\Escritorio\Programacion\Programas\Compu2324\Obligatorio1\posiciones.dat', 'r') as f:
    coordenadas = [[list(map(float, line.split(','))) for line in block.split('\n') if line] for block in f.read().split('\n\n')]
    
v.pop()
coordenadas.pop()

filas = len(v)

# Calcular la energía cinética.
E_cin = np.zeros((n, int(filas)))


for k in range(filas):
    for i in range(n):
            E_cin[i][k] = 0.5 * m_rees[i] * np.linalg.norm(v[k][i])**2
       

#Calcular energía potencial.
E_pot = np.zeros((n, int(filas)))


for k in range(len(coordenadas)):
    for i in range(n):
        for j in range(n):
            if i != j:
                E_pot[i][k] += (-m_rees[i]*m_rees[j])/(np.linalg.norm(np.array(coordenadas[k][i])-np.array(coordenadas[k][j]))*2)
        E_pot[i][k] += (-m_rees[i])/(np.linalg.norm(np.array(coordenadas[i]))*2)
                
# Inicializar E_planeta array
E_planeta = np.zeros((n, int(filas)))

for i in range(n):
    for j in range(int(filas)):
        E_planeta[i][j] = E_cin[i][j] + E_pot[i][j]


# Graficar la energía cinética            
labels = ['Mercurio', 'Venus', 'Tierra', 'Marte', 'Jupiter', 'Saturno', 'Urano', 'Neptuno']




for i in range(n):
    plt.figure(i)  # Crear una nueva figura para cada conjunto de datos

    # Graficar los datos
    plt.plot(E_cin[i], label='Energía cinética ' + labels[i])
    plt.plot(E_pot[i], label='Energía potencial ' + labels[i])
    plt.plot(E_planeta[i], label='Energía total ' + labels[i])

#plt.plot(E_total[0], label='Energía total ' )

# Añadir título y etiquetas
plt.xlabel('Index')
plt.ylabel('Energía')
plt.legend()

# Mostrar la gráfica
plt.show()
