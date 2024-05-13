import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de los archivos
data1 = np.loadtxt('energia_cinetica.dat')
data2 = np.loadtxt('energia_potencial.dat')
data3 = np.loadtxt('energia_part.dat')
data4 = np.loadtxt('velocidades_part.dat', delimiter=',')
data5 = data4[:,0]
data6 = data4[:,1]


# Crear una figura y un eje
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()



# Graficar los datos
ax.plot(data1, color='blue', label='Energía Cinética')
ax.plot(data2, color='red', label='Energía Potencial')
ax.plot(data3, color='green', label='Energía Total')


ax2.plot(data5, color='blue', label='Velociades x')
ax2.plot(data6, color='red', label='Velociades y')

# Añadir una leyenda
ax.legend()
ax.set_title('Energías en el tiempo')
ax.set_xlabel('Iteraciones')
ax.set_ylabel('Energía')

# Añadir una leyenda velociades
ax2.legend()
ax2.set_title('Velocidades en el tiempo')
ax2.set_xlabel('Iteraciones')
ax2.set_ylabel('Velocidad') 


# Mostrar el gráfico
plt.show()