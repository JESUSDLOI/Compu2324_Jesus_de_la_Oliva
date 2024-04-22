import matplotlib.pyplot as plt
import numpy as np

# Crear un array de valores x
x = np.linspace(-10, 10, 20)

# Calcular los valores y correspondientes
y = x**2

# Crear el gráfico
plt.plot(x, y)

# Añadir títulos y etiquetas
plt.title("Gráfico de y = x^2")
plt.xlabel("x")
plt.ylabel("y")

# Mostrar el gráfico
print(x)
print(y)
plt.show()
