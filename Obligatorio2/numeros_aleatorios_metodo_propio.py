import numpy as np  


def split_number(n):
    return [int(digit) for digit in str(n)]

m = 1
digits = split_number(m)
v = np.zeros(m)
for i in range(m):
   v[i]= np.random.choice(digits)
#with open('C:/Users/jesol/OneDrive/Escritorio/Compu/Compu2324_Jesus_de_la_Oliva/Obligatorio2/random_numbers.txt', 'w') as f:
   # for i in v:
   #     f.write(str(i) + ' ')    
print(v)