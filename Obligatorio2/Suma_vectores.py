import numpy as np

# Definir tamaño y valores posibles de los vectores
n=3
a=1
b=10

def suma_vectores(a,b,n):    
    v1 = np.random.randint(a,b,n)
    v2 = np.random.randint(a,b,n)
    v3 = np.add(v1,v2)

    return v1,v2,v3
    
v1,v2,v3 = suma_vectores(a,b,n)

print(v1)
print(v2)
print(v3)   
