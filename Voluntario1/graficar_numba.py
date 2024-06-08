import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt('part_tiempo.dat', delimiter=',')
data1 = np.loadtxt('part_tiempo_no_numba.dat', delimiter=',')

part = data[:,0]
tiempo = data[:,1]

part1 = data1[:,0]
tiempo1 = data1[:,1]

plt.figure(figsize=(10, 6))  # Crea una nueva figura con un tamaño específico
plt.plot(part, tiempo, 'o-', label='Simulaciones Numba')  # Grafica 'part' en el eje x y 'tiempo' en el eje y
plt.plot(part1, tiempo1, 'o-', label='Simulaciones sin Numba')
plt.xlabel('Partículas', fontsize=20)  # Etiqueta del eje x
plt.ylabel('Tiempo', fontsize=20)  # Etiqueta del eje y
plt.title('Gráfica de Part vs Tiempo', fontsize=30)  # Título de la gráfica
plt.xticks(fontsize=15)  # Tamaño de la fuente de los números del eje x
plt.yticks(fontsize=15)  # Tamaño de la fuente de los números del eje y
plt.legend(fontsize=20)  # Muestra la leyenda
plt.grid(True)  # Muestra la cuadrícula
plt.show()  # Muestra la gráfica