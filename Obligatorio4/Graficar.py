import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de los archivos
data1 = np.loadtxt('H_h_fija.dat')
data2 = np.loadtxt('H_h_ajustada.dat')

# Crear una figura y un eje
fig, ax = plt.subplots()

# Graficar los datos
ax.plot(data1, color='blue', label='Paso h fijo')
ax.plot(data2, color='red', label='Paso h ajustado')

# Añadir una leyenda
ax.legend()
ax.set_title('Comparación de los valores de H con h fijo y h ajustado')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('H (J)')

# Mostrar el gráfico
plt.show()