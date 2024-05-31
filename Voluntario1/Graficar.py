import matplotlib.pyplot as plt
import numpy as np

simulacion = 3

# Cargar los datos de los archivos
data1 = np.loadtxt('energia_cinetica' + str(simulacion) + '.dat')
data2 = np.loadtxt('energia_potencial' + str(simulacion) + '.dat')
data3 = np.loadtxt('energia_part' + str(simulacion) + '.dat')
data4 = np.loadtxt('velocidades_part' + str(simulacion) + '.dat', delimiter=',')
data5 = data4[:,0]
data6 = data4[:,1]
data4 = np.linalg.norm(data4, axis=1)
data7 = np.loadtxt('presion_temp.dat', delimiter=',')
data8 = np.loadtxt('presion' + str(simulacion) + '.dat')



# Crear una figura y un eje
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

numero_datos = len(data4)
v1 = np.linspace(-2, 2, 1000)
v2 = np.linspace(0, 3, 1000)

if data7.ndim == 1:
 T = data7[1]
else:
 T = data7[simulacion][1]
v_cua2 = v2**2
v_cua1 = v1**2

# Graficar los datos de la energía del sistema en el tiempo
ax.plot(data1, color='blue', label='Energía Cinética')
ax.plot(data2, color='red', label='Energía Potencial')
ax.plot(data3, color='green', label='Energía Total')

#Histograma de velocidades
ax2.hist(data4, bins=100, color='green', label='Velocidades')
ax2.plot(v2, (1/T)*v2*np.exp(-v_cua2/(2*T))*numero_datos/10)
ax3.hist(data5, bins=100, color='blue', label='Velociades x')
ax3.plot(v1, np.sqrt(1/(2*np.pi*T))*np.exp(-v_cua1/(2*T))*numero_datos/10)
#ax2.hist(data6, bins=100, color='red', label='Velociades y')

tiempo = np.linspace(0, len(data8), len(data8))
ax4.scatter(tiempo, data8, color='red', label='Presión')

if data7.ndim == 1:
    ax5.scatter(data7[0], data7[1],  color='blue')

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']  # Define your colors here
labels = ['Velocidad modulo 0', 'Velocidad modulo 1', 'Velocidad modulo 2', 'Velocidad modulo 3', 'Velocidad modulo 4', 'Velocidad modulo 5', 'label7']  # Define your labels here

for i in range(len(data7)):
    ax5.scatter(data7[i, 0], data7[i, 1], color=colors[i % len(colors)], label=labels[i % len(labels)])

# Supongamos que tus datos están en data7
x = data7[:, 0]
y = data7[:, 1]

# Realizar el ajuste lineal
coeffs = np.polyfit(x, y, 1)

# coeffs es una lista de coeficientes, donde el primer elemento es la pendiente y el segundo es la intersección y
slope, intercept = coeffs

# Ahora puedes usar la pendiente y la intersección y para trazar la línea de ajuste
x_fit = np.linspace(x.min(), x.max(), 100)  # 100 puntos entre el mínimo y el máximo x
y_fit = slope * x_fit + intercept
y_pred = slope * x + intercept
# Calcular los residuos
residuos = y - y_pred

# Calcular el error cuadrático medio (MSE)
mse = np.mean(residuos**2)

# Calcular el chi cuadrado
chi_cuadrado = 100 - np.sum((residuos / y_pred)**2)


ax5.plot(x_fit, y_fit, color='black', label= 'Ajuste linal de los puntos')  # línea de ajuste en rojo
ax5.text(0.8, 0.3, f'y = {slope:.2f}x + {intercept:.2f}\nMSE = {mse:.2f}\nChi^2 = {chi_cuadrado:.2f}', transform=ax5.transAxes, fontsize=12, verticalalignment='top')  # añadir texto con los valores de la pendiente y la intersección
ax5.legend()
ax5.set_title('Relación lineal entre la presión la temperatura', fontsize=20)
ax5.set_xlabel('Presión unidades reescaladas', fontsize=15)
ax5.set_ylabel('Temperatura unidades reescaladas', fontsize=15)
ax5.tick_params(axis='both', labelsize=12)




# Añadir una leyenda
ax.legend()
ax.set_title('Energías en el tiempo', fontsize=20)
ax.set_xlabel('Iteraciones', fontsize=15)
ax.set_ylabel('Energía', fontsize=15)


# Añadir una leyenda velociades
ax2.legend()
ax2.set_title('Velocidades en el tiempo')
ax2.set_xlabel('Velocidad')
ax2.set_ylabel('Frecuencia') 


# Mostrar el gráfico
plt.show()