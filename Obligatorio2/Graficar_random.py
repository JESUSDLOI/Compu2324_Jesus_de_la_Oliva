import matplotlib.pyplot as plt
from collections import Counter

#Representar datos aleatorios del documento random_numbers.txt
data = open('C:/Users/jesol/OneDrive/Escritorio/Compu/Compu2324_Jesus_de_la_Oliva/Obligatorio2/random_numbers.txt', 'r').read().split('\n')
# Calcular frecuencias
frequencies = Counter(data)

# Ordenar por frecuencias
sorted_data = sorted(frequencies.items(), key=lambda x: x[1])

# Separar en dos listas
keys, values = zip(*sorted_data)

# Plotear
plt.bar(keys, values, color='c', edgecolor='black')
plt.title('Histograma de números aleatorios')
plt.grid(True)
plt.yticks(range(0, 10, 1))
plt.show()
