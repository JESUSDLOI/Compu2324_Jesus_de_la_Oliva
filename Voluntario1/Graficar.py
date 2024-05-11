import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de los archivos
data1 = np.loadtxt('energia_cinetica.dat')
data2 = np.loadtxt('energia_potencial.dat')

# Crear una figura y un eje
fig, ax = plt.subplots()

# Graficar los datos
ax.plot(data1, color='blue', label='Cáculos Portatil')
ax.plot(data2, color='red', label='Cálculos Joel')

# Añadir una leyenda
ax.legend()
ax.set_title('Tiempo de cálculo por planeta añadido')
ax.set_xlabel('Número de planetas')
ax.set_ylabel('Tiempo de cálculo (s)')

# Mostrar el gráfico
plt.show()