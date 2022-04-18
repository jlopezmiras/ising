import numpy as np
import math
from numba import njit
import timeit



@njit
def ising(temp,espines,steps,fout=None):
    
    # Dimensiones de la red NxN
    N = np.shape(espines)[0]

    # Vector que almacena la configuración de espines en todos los pasos montecarlo
    Stot = np.empty((steps,N,N))

    # Abro archivo de texto
    #f = open(fout, "w")

    for pasoMc in range(steps):
        # Un paso Monte Carlo son N² iteraciones
        for i in range(N*N):

            n,m = np.random.randint(0,N,size=2)

            if n!=0 and m!=0 and n!=N-1 and m!=N-1:
                deltaE = 2*espines[n,m]*(espines[n+1,m] + espines[n-1,m] + espines[n,m+1] + espines[n,m-1])
            else: 
                deltaE = 2*espines[n,m]*(espines[(n+1)%N,m] + espines[(n-1)%N,m] + espines[n,(m+1)%N] + espines[n,(m-1)%N])

            p = min(1,math.exp(-deltaE/temp))
            aleatorio = np.random.random()
            if aleatorio<p:
                espines[n,m] = -espines[n,m]

        Stot[pasoMc] = espines
        #np.savetxt(f,S,delimiter=', ')
        #f.write('\n')
        
    return Stot



espines = np.random.choice([-1,1],size=(64,64))

T = 0.5
steps = 200

fout = "ising_data.dat"

start = timeit.default_timer()

Stot = ising(0.1,espines,200)
print("A escribir!")

f = open(fout, "w")
for i in range(len(Stot)):
    np.savetxt(f,Stot[i],delimiter=', ')
    f.write('\n')
f.close()

stop = timeit.default_timer()
print("Time:", str(stop-start))

