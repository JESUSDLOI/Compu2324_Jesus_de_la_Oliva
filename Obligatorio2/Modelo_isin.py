import numpy as np 
from numpy import random
import scipy as sp
from numba import jit

#Matriz aleatoria entre estado 1 y -1
def mtrz_aleatoria(N):
    matriz = np.random.choice([-1,1],(N,N))
    return matriz



#Matriz de Ising
def ising_model(N, T):
    #Matriz de Ising
    matriz = mtrz_aleatoria(N)
    print(matriz)

    #Condiciones de contorno periódicas
    matriz = np.pad(matriz, pad_width=2, mode='wrap')
    print(matriz)
    matriz = np.delete(matriz, [1, N+1], axis=0)
    matriz = np.delete(matriz, [1, N+1], axis=1)
    print(matriz)

    #Archivo de datos
    with open('ising_data.dat', 'w') as file:

        for k in range(N**2):
            #Iteración sobre la matriz
            i = np.random.randint(1,N+1)
            j = np.random.randint(1,N+1)

            #Calculo de la variación de la energía
            delta_E = 2*matriz[i,j]*(matriz[(i+1),j] + matriz[i,(j+1)] + matriz[(i-1),j] + matriz[i,(j-1)])
            
            #Evaluar probabilidad de cambio
            p = np.min([1, np.exp(-2*delta_E/T)])

            #Número aleatorio entre  para comparar con la probabilidad
            r = np.random.rand(1,1)

            if r < p:
                matriz[i,j] = -matriz[i,j]  
            
            submatriz = matriz[1:N+1, 1:N+1]    

            np.savetxt(file, submatriz, fmt='%d', delimiter=',') 
            file.write('\n')

    file.close()

ising_model(50, 0.1)

