import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de los archivos
data1 = np.loadtxt('energia_cinetica.dat')
data2 = np.loadtxt('energia_potencial.dat')
data3 = np.loadtxt('energia_part.dat')
data4 = np.loadtxt('velocidades_part.dat', delimiter=',')
data5 = data4[:,0]
data6 = data4[:,1]
data4 = np.linalg.norm(data4, axis=1)
data7 = np.loadtxt('presion_temp.dat', delimiter=',')



# Crear una figura y un eje
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

v1 = np.linspace(-2, 2, 1000)
v2 = np.linspace(0, 3, 1000)
T = 12.574962939932645
v_cua2 = v2**2
v_cua1 = v1**2

# Graficar los datos de la energía del sistema en el tiempo
ax.plot(data1, color='blue', label='Energía Cinética')
ax.plot(data2, color='red', label='Energía Potencial')
ax.plot(data3, color='green', label='Energía Total')

#Histograma de velocidades
ax2.hist(data4, bins=100, color='green', label='Velocidades')
ax2.plot(v2, (1/T)*v2*np.exp(-v_cua2/(2*T))*3000)
ax3.hist(data5, bins=100, color='blue', label='Velociades x')
ax3.plot(v1, np.sqrt(1/(2*np.pi*T))*np.exp(-v_cua1/(2*T))*3000)
#ax2.hist(data6, bins=100, color='red', label='Velociades y')

ax5.plot(data7.iloc[:, 0], data7.iloc[:, 1],  color='blue')


# Añadir una leyenda
ax.legend()
ax.set_title('Energías en el tiempo')
ax.set_xlabel('Iteraciones')
ax.set_ylabel('Energía')

# Añadir una leyenda velociades
ax2.legend()
ax2.set_title('Velocidades en el tiempo')
ax2.set_xlabel('Velocidad')
ax2.set_ylabel('Frecuencia') 


# Mostrar el gráfico
plt.show()