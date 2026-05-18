# n=1: aprox 0.015eV
# n=2: aprox. 0.06 eV
# n=3: aprox. 0.135 eV

import numpy as np
import matplotlib.pyplot as plt

#Constante fizice
m = 9.1e-31         # masa electronului
hbar = 1.054e-34    # J*s - constanta Planck (h_bar = h / 2pi)
eV_to_J = 1.6e-19   # factor de conversie din eV in Joule

#Datele problemei 
a = 5e-9            # latimea gropii = 5 nm
N = 100             # numarul de noduri
dx = a / (N + 1)    # pasul retelei spatiale (distanta dintre doua noduri consecutive)

# Coeficientul din fata derivatei a doua in Ecuatia lui Schrodinger
# H = -(hbar^2 / 2m) * d^2/dx^2
C = (hbar**2) / (2 * m * dx**2)

# Initializam matricea Hamiltoniana (H) cu zerouri
H = np.zeros((N, N))

# Construim matricea tridiagonala pentru metoda diferentelor finite
# Pe diagonala principala punem 2*C, iar pe cele de langa -1C
for i in range(N):
    H[i, i] = 2 * C
    if i > 0:
        H[i, i-1] = -C
    if i < N - 1:
        H[i, i+1] = -C

# Rezolvam ecuatia matriceala de valori proprii: H * psi = E * psi
# Folosesc eigh din numpy pentru ca stim ca matricea Hamiltoniana este simetrica (hermitica), si asta ne da direct eigenvalues sortate crescator
valori_proprii, vectori_proprii = np.linalg.eigh(H)

# Valorile proprii reprezinta energiile in Jouli, le convertim in eV:
E_num_eV = valori_proprii / eV_to_J

# Extragem primele 10 energii calculate numeric
E_num_primele = E_num_eV[:10]

# La seminar am calculat deja ca E_n in [eV] e aprox 0.015 * n^2 pentru a=5 nm
n = np.arange(1, 11)   # luam primele 10 nivele cuantice
E_exact = 0.015 * n**2 

# Printam o mica verificare in consola ca sa vedem daca da bine inainte de grafic
print("Comparatie pentru primele 3 nivele energetice:")
print("Exact (eV)  |  Numeric (eV)")
for i in range(3):
    print(f"{E_exact[i]:.4f}      |  {E_num_primele[i]:.4f}")

# grafice
plt.figure(figsize=(8, 6))

# Plotez curba analitica ca o linie continua
plt.plot(n, E_exact, 'r-', linewidth=2, label='Valori exacte (analitic - rel. 14)')

# Plotez valorile obtinute numeric cu puncte albastre
plt.plot(n, E_num_primele, 'bo', markersize=6, label='Valori numerice (diferențe finite)')

plt.title('Tema 1: Nivelele de energie într-o groapă infinită de potențial ($a=5$ nm)')
plt.xlabel('Numărul cuantic principal ($n$)')
plt.ylabel('Energia $E_n$ [eV]')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

b = 5  # luam in nm direct pentru axa X

# Definim un sir de valori pentru axa x, de la 0 la 5 nm
x = np.linspace(0, b, 200)

normare = np.sqrt(2 / b)

# Definim functiile de unda pentru primele 3 stari (n=1, 2, 3)
psi_1 = normare * np.sin(1 * np.pi * x / b)
psi_2 = normare * np.sin(2 * np.pi * x / b)
psi_3 = normare * np.sin(3 * np.pi * x / b)

# Cream un grafic nou pentru functiile de unda
plt.figure(figsize=(10, 6))

# Plotez cele trei functii
plt.plot(x, psi_1, 'r-', linewidth=2, label='Starea fundamentală ($n=1$)')
plt.plot(x, psi_2, 'g--', linewidth=2, label='Prima stare excitată ($n=2$)')
plt.plot(x, psi_3, 'b-.', linewidth=2, label='A doua stare excitată ($n=3$)')

#asta e un check ca sa vedem unde trece functia de unda prin 0
plt.axhline(0, color='black', linewidth=1)

plt.title('Tema 2: Funcțiile de undă $\psi_n(x)$ pentru primele 3 stări ($b=5$ nm)')
plt.xlabel('Poziția $x$ [nm]')
plt.ylabel('Amplitudinea funcției de undă $\psi(x)$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()