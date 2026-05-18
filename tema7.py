import numpy as np
import matplotlib.pyplot as plt

# Parametrii de discretizare 
N = 1000
dx = 0.01
x = np.arange(0, N + 1) * dx   # Coordonata x de la 0 la 10 

# Discretizarea energiei
E_min, E_max, dE = 0.0, 15.0, 0.003
energies = np.arange(E_min, E_max + dE, dE)

# Potentialul oscilatorului adimensional: W(x) = x^2 / 4
V = (x**2) / 4.0

def numerov_integrate(E, is_even=True):
    """Integreaza ecuatia Schrödinger folosind algoritmul Numerov"""
    phi = np.zeros(N + 1)
    
    # Conditii la limita in origine
    if is_even:
        phi[0] = 1.0   # Solutiile pare in s7 au pornit cu phi[0]=1, phi[1]=1 la derivata nula
        phi[1] = 1.0   # Corectat conform conventiei din s7.pdf pag 4: phi_i,0 := 1, phi_i,1 := 1
    else:
        phi[0] = 0.0   # Solutiile impare pornesc din 0
        phi[1] = dx    # Panta initiala

    # Factorii g si f din algoritm
    g = E - V
    f = 1.0 + (dx**2 / 12.0) * g

    # Integrare pas cu pas
    for j in range(1, N):
        phi[j+1] = ((12.0 - 10.0 * f[j]) * phi[j] - f[j-1] * phi[j-1]) / f[j+1]
        
        # Guard impotriva divergentelor extreme (Metoda Tirului)
        if abs(phi[j+1]) > 1e5:
            return phi[:j+2], j+1
            
    return phi, N

#Metoda Tirului pentru determinarea starii fundamentale (n=0, E=0.5)
# Cautam energia unde functia tinde la 0 la infinit (x -> 10)
E_test_1 = 0.490
E_test_2 = 0.510

phi_1, _ = numerov_integrate(E_test_1, is_even=True)
phi_2, _ = numerov_integrate(E_test_2, is_even=True)

# Plot pentru vizualizarea metodei tirului
plt.figure(figsize=(8, 5))
plt.plot(x[:len(phi_1)], phi_1, 'r-', label=f'E = {E_test_1} (Diverge la +inf)')
plt.plot(x[:len(phi_2)], phi_2, 'b--', label=f'E = {E_test_2} (Diverge la -inf)')
plt.axhline(0, color='black', lw=0.5)
plt.xlabel('$x_j$')
plt.ylabel('$\phi_j$')
plt.title('Metoda Tirului: Cautarea valorii proprii pentru n=0 ($E_{exact}=0.5$)')
plt.ylim(-100, 100)
plt.legend()
plt.grid(True)
plt.show()