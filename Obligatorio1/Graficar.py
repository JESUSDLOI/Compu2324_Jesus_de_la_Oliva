import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de los archivos
data1 = np.loadtxt('Tiempo_planetas.dat', delimiter=',')
data2 = np.loadtxt('Tiempo_planetas_.dat', delimiter=',')

# Crear una figura y un eje
fig, ax = plt.subplots()

# Graficar los datos
ax.plot(data1[:, 0], data1[:, 1], color='blue', label='Cáculos Portatil')
ax.plot(data2[:, 0], data2[:, 1], color='red', label='Cálculos Joel')

# Añadir una leyenda
ax.legend()
ax.set_title('Tiempo de cálculo por planeta añadido')
ax.set_xlabel('Número de planetas')
ax.set_ylabel('Tiempo de cálculo (s)')

# Mostrar el gráfico
plt.show()