import numpy as np
import matplotlib.pyplot as plt

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

# Linii ajutatoare (axa OX in y=0) pentru a vedea cum functia trece prin 0
plt.axhline(0, color='black', linewidth=1)

plt.title('Tema 4: Funcțiile de undă $\psi_n(x)$ pentru primele 3 stări ($b=5$ nm)')
plt.xlabel('Poziția $x$ [nm]')
plt.ylabel('Amplitudinea funcției de undă $\psi(x)$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()