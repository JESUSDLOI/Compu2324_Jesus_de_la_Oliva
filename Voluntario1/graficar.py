import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer los datos del archivo
data = pd.read_csv('aceleraciones_part.dat', delimiter=",")
data = data.values

# Asegurarse de que los datos son una serie de vectores
for i in range(data):
    x = data[3*i]
    y = data[3*i + 1]
    z = data[3*i + 2]

# Graficar los datos
plt.plot(x, y, z)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aceleraciones de las partículas')
plt.show()
    
