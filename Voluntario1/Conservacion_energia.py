import numpy as np
import matplotlib.pyplot as plt
from numba import jit


#Numero de particulas
n = 20

l = 20

# Leer las velocidades de un archivo de texto
with open('velocidades_part.dat', 'r') as f:
    v = [[list(map(float, line.split(','))) for line in block.split('\n') if line] for block in f.read().split('\n\n')]

with open('posiciones_part.dat', 'r') as f:
    coordenadas = [[list(map(float, line.split(','))) for line in block.split('\n') if line] for block in f.read().split('\n\n')]
    
v.pop()
coordenadas.pop()
filas = len(v)
v = np.array(v)
coordenadas = np.array(coordenadas)

# Calcular la energía cinética.
@jit(nopython=True, fastmath=True, cache=True)
def energia_cinetica(v, n, filas):
    E_cin = np.zeros((n, int(filas)))
    for k in range(filas):
        for i in range(n):
            E_cin[i][k] = (np.linalg.norm(v[k][i])**2)/ 2
    return E_cin


#Función para calcular la distancia entre dos partículas.
@jit(nopython=True, fastmath=True, cache=True)
def distancia_condiciones(posicion, i, j, l):
    resta = np.zeros(2)
    
    resta[0] = np.abs(posicion[i][0] - posicion[j][0])
    if resta[0] > l/2:
        resta[0] = np.abs(resta[0] - l)

    resta[1] = np.abs(posicion[i][1] - posicion[j][1])
    if resta[1] > l/2:
        resta[1] = np.abs(resta[1] - l)
    
    distancia = np.sqrt(resta[0]**2 + resta[1]**2)
    return distancia


# Calcular la energía potencial.
@jit(nopython=True, fastmath=True, cache=True)
def energia_potencial(coordenadas, n, l):
    E_pot = np.zeros((n, int(filas)))
    rango = len(coordenadas)
    for k in range(rango-1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    distancia = distancia_condiciones(coordenadas[k+1], i, j, l)
                    if distancia > 0.001 and distancia < 3:
                        E_pot[i][k+1] += 4*(1/distancia**12 - 1/distancia**6)
    return -E_pot

# Calcular la energía total.
@jit(nopython=True, fastmath=True, cache=True)
def energia_total(E_cin, E_pot, n, filas):
    E_total = np.zeros((int(filas)))
    E_planeta = np.zeros((n, int(filas)))
    for i in range(n):
        for j in range(int(filas)):
            E_planeta[i][j] = E_cin[i][j] + E_pot[i][j]
            E_total[j] += E_planeta[i][j]
    return E_total

E_cin = energia_cinetica(v, n, filas)   
E_pot = energia_potencial(coordenadas, n, l)
E_total = energia_total(E_cin, E_pot, n, filas)


# Graficar la energía cinética            
labels = np.array(['partícula ' + str(i) for i in range(n)])


 
#plt.figure(i)
# Graficar los datos
'''
plt.plot(E_cin[i], label='Energía cinética ' + labels[i])
plt.plot(E_pot[i], label='Energía potencial ' + labels[i])
plt.plot(E_planeta[i], label='Energía total ' + labels[i])
'''
plt.plot(E_total, label='Energía total')

# Añadir título y etiquetas
plt.xlabel('Index')
plt.ylabel('Energía')
plt.legend()

# Mostrar la gráfica
plt.show()