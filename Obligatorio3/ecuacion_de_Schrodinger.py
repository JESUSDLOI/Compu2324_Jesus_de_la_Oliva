import numpy as np
import time
from numba import jit


t_o = time.time()

#Iteraciones
iteraciones = 10000

# Parametros iniciales
N = 400
nciclos = N/4
lamb = 1.2

# Generar s, k0, Vj, Φj_0 and alpha
k0 = 2 * np.pi * nciclos/ N
s = 1 / (4*k0**2)

#Inicializar las variables
Vj = np.zeros(N)
Oj_n = np.zeros(N, dtype=complex)    
Oj_0 = np.zeros(N, dtype=complex)
alpha = np.zeros(N, dtype=complex)
beta = np.zeros(N, dtype=complex)
Xj_n = np.zeros(N, dtype=complex)
phi = np.zeros(N)

#Calcular potencial, Vj
@jit(nopython=True, fastmath=True, cache=True)
def calc_Vj(N, lamb, k0, Vj):
    V2 = np.zeros(N)
    for i in range(N):
        if i >= 2*N/5 and i <= 3*N/5:
            Vj[i] = lamb * k0**2
        else:
            Vj[i] = 0
        V2[i] = abs(Vj[i])**2
    return Vj

#Función onda inicial
@jit(nopython=True, fastmath=True, cache=True)
def onda_inicial(N, k0, Oj_0, phi):
    for i in range(N):
        Oj_0[i] = np.exp(1j * k0 * i)*np.exp(-8*(4*i-N)**2 / N**2)
    
    Oj_0[N-1] = Oj_0[0] = 0
    for i in range(N):
        phi[i] = np.abs(Oj_0[i])**2     
    return Oj_0, phi

# Calcular alpha
@jit(nopython=True, fastmath=True, cache=True)
def calc_alpha(s, Vj, alpha, N):
    for i in range(N-1, 0, -1):
        alpha[i-1] = -1/((-2 + (2j/s) - Vj[i])+(alpha[i]))
    return alpha

# Calcular beta
@jit(nopython=True, fastmath=True, cache=True)
def calc_beta(s, Oj_0, alpha, beta, Vj, N):
    for i in range(N-1, 0, -1):
        beta[i-1] = ((4j*Oj_0[i]/s)-beta[i])/((-2 + 2j/s - Vj[i])+(alpha[i]))
    return beta

# Calcular Xj_n
@jit(nopython=True, fastmath=True, cache=True)
def calc_Xj_n(Xj_n, alpha, beta, N):
    for i in range(N-1):
        Xj_n[i+1] = alpha[i]*Xj_n[i] + beta[i]
    return Xj_n

# Calcular Oj_n
@jit(nopython=True, fastmath=True, cache=True)
def calc_Oj_n(Xj_n, Oj_n, N, phi):
    for i in range(N):
        Oj_n[i] = Xj_n[i] - Oj_n[i]
        phi[i] = np.abs(Oj_n[i])**2
    return Oj_n, phi

# Calcular la función de onda temporal
def funcion_onda_temporal(N, k0, Oj_0, iteraciones, s, Vj, alpha, beta, Xj_n, Oj_n, phi, file, file_2):
    
    norma = 0
    Oj_0, phi = onda_inicial(N, k0, Oj_0, phi)
    Vj = calc_Vj(N, lamb, k0, Vj)
    alpha = calc_alpha(s, Vj, alpha, N)

    for j in range(iteraciones):
        beta = calc_beta(s, Oj_0, alpha, beta, Vj, N)
        Xj_n = calc_Xj_n(Xj_n, alpha, beta, N)
        Oj_n, phi = calc_Oj_n(Xj_n, Oj_n, N, phi)
        Oj_0 = Oj_n
        norma = np.sqrt(sum(phi))
        phi = phi / norma
        file_2.write(f"{j}, {norma}\n")
        for i in range(N):
            file.write(f"{i}, {phi[i]}, {Vj[i]}\n")
        file.write("\n")
        
        
file=open('schrodinger_data.dat','w')    
file_2=open('norma_data.dat','w')
    
#Llamamos a las funciones
funcion_onda_temporal(N, k0, Oj_0, iteraciones, s, Vj, alpha, beta, Xj_n, Oj_n, phi, file, file_2)

file.close()
file_2.close()

t_f = time.time()

print("Tiempo de ejecución: ", t_f - t_o)
        
        

