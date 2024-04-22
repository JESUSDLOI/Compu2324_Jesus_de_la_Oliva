import numpy as np 
from numpy import random

# Definir tamaño y valores posibles de los vectores
n=100000
a=1
b=100

def suma_vectores(v1,v2):
    v3 = np.add(v1,v2)

    return v3

def vectores_aleatorios(a,b,n):
    v1 = np.random.randint(a,b,n)
    v2 = np.random.randint(a,b,n)

    return v1,v2
    
v1,v2 = vectores_aleatorios(a,b,n)
v3 = suma_vectores(v1,v2)

with open('C:/Users/jesol/OneDrive/Escritorio/Compu/Compu2324_Jesus_de_la_Oliva/Obligatorio2/random_numbers.txt', 'w') as f:
    for i in v2:
        f.write(str(i) + ' ')

#print(v1)
#print(v2)
#print(v3)   
