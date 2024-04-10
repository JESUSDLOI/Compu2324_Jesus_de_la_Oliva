import matplotlib.pyplot as plt
from collections import Counter

#Numero de planetas
n = 6

# Leer los datos del archivo
with open('posiciones.dat', 'r') as f:
    data = [line.strip() for line in f]


# Split each line by comma and convert to float
coordinates = [list(map(float, line.split(','))) for line in data if line]

# Separate into x and y coordinates
x, y = zip(*coordinates)

#Numero de filas
filas = len(x)


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange']

k = 0

while k < len(x):
    for i in range(n):
        plt.scatter(x[k], y[k], color=colors[i])
        k += 1
    
# Leyenda
labels = ['Sol', 'Mercurio', 'Venus', 'Tierra', 'Marte', 'Jupiter', 'Saturno', 'Urano', 'Neptuno']

# Crear los parches
patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], 
            label="{:s}".format(labels[i]) )[0]  for i in range(n) ]

# Añadir la leyenda
plt.legend(handles=patches, loc='upper left', ncol=2, numpoints=1 )

# Mostrar el gráfico
plt.title('Scatter plot of coordinates')
plt.grid(True)
plt.autoscale(tight=True)
plt.show()
