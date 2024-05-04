import numpy as np
from scipy.integrate import odeint
import time

t_o = time.time()

#Iteraciones
iteraciones = 100

# Parametros iniciales
N = 10000
nciclos = N/4
λ = 0.5

# Generar s, k0, Vj, Φj_0 and alpha
k0 = 2 * np.pi / 4
s = 1 / (4*k0**2)

#Inicializar las variables
Vj = np.zeros(N, dtype=complex)
Oj_n = np.zeros(N, dtype=complex)    
Oj_0 = np.zeros(N, dtype=complex)
alpha = np.zeros(N, dtype=complex)
beta = np.zeros(N, dtype=complex)
Xj_n = np.zeros(N, dtype=complex)

#Calcular potencial, Vj
def calc_Vj(N, λ, k0, Vj):
    for i in range(N):
        if i >= 2*N/5 and i <= 3*N/5:
            Vj[i] = λ * k0**2
        else:
            Vj[i] = 0
    return Vj

#Función onda inicial
def onda_inicial(N, k0, Oj_0):
    for i in range(1, N-1):
        Oj_0[i] = np.exp(-1j * k0 * i)*np.exp((-8*i-N)**2 / N**2)
    Oj_0[0] = 0
    Oj_0[N-1] = 0
    return Oj_0

# Calcular alpha
def calc_alpha(s, Vj, alpha, N):
    for i in range(N-1, 0, -1):
        alpha[i-1] = -1/((-2 + 2j/s - Vj[i])+(alpha[i]))
    return alpha

# Calcular beta
def calc_beta(s, Oj_0, alpha, beta, Vj, N):
    for i in range(N-1, 0, -1):
        beta[i-1] = (4j*Oj_0[i]-beta[i])/((-2 + 2j/s - Vj[i])+(alpha[i]))
    return beta

# Calcular Xj_n
def calc_Xj_n(Xj_n, alpha, beta, N):
    for i in range(N-1):
        Xj_n[i+1] = alpha[i]*Xj_n[i] + beta[i]
    return Xj_n

# Calcular Oj_n
def calc_Oj_n(Xj_n, Oj_n, file, j, N):
    for i in range(N):
        Oj_n[i] = Xj_n[i] - Oj_n[i]
        file.write(f"{j}, {Oj_n[i].real}, {Oj_n[i].imag}\n".format(i, Oj_n[i].real, Oj_n[i].imag))
        
    return Oj_n

# Calcular la función de onda temporal
def funcion_onda_temporal(N, k0, Oj_0, iteraciones, s, Vj, alpha, beta, Xj_n, Oj_n):
    
    onda_inicial(N, k0, Oj_0)
    Vj = calc_Vj(N, λ, k0, Vj)
    
    file=open('schrodinger_data.dat','w')
    
    for j in range(iteraciones):
        alpha = calc_alpha(s, Vj, alpha, N)
        beta = calc_beta(s, Oj_0, alpha, beta, Vj, N)
        Xj_n = calc_Xj_n(Xj_n, alpha, beta, N)
        Oj_n = calc_Oj_n(Xj_n, Oj_n, file, j, N)
        Oj_0 = Oj_n
            
        
    file.close()
    
#Llamamos a las funciones
funcion_onda_temporal(N, k0, Oj_0, iteraciones, s, Vj, alpha, beta, Xj_n, Oj_n)

t_f = time.time()

print("Tiempo de ejecución: ", t_f - t_o)
        
        

