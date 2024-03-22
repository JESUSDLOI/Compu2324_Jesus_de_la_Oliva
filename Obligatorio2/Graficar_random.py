import matplotlib.pyplot as plt
from collections import Counter


# Read data from the file
data = open('C:/Users/jesol/OneDrive/Escritorio/Compu/Compu2324_Jesus_de_la_Oliva/Obligatorio1/posiciones.txt', 'r').read().split('\n')

# Split each line by comma and convert to float
coordinates = [list(map(float, line.split(','))) for line in data if line]

# Separate into x and y coordinates
x, y = zip(*coordinates)

# Plot the data
plt.scatter(x, y)
plt.title('Scatter plot of coordinates')
plt.grid(True)
plt.autoscale(tight=True)
plt.show()
