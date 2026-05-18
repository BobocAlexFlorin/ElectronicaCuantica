import numpy as np
import matplotlib.pyplot as plt

# 2. Constante fizice & Parametrii sistemului
e_charge = 1.602176634e-19  # Sarcina electronului (C)
h_plank = 6.62607015e-34    # Constanta lui Planck (J*s)
hbar = h_plank / (2 * np.pi)
m_electron = 9.1093837e-31  # Masa electronului (kg)

# Inaltimile barierelor de potential pentru fiecare domeniu (in eV)
V = np.array([0, 0.2, 0, 0.2, 0]) 
# Latimile domeniilor (in nm) - doar domeniile interioare conteaza la transport (1, 2, 3)
# Domeniul 0 (stanga) si Domeniul 4 (dreapta) sunt semi-infinite
widths_nm = np.array([5, 5, 5]) 
widths_m = widths_nm * 1e-9

# Factorul de conversie t (folosit pentru vectorul de unda k)
t_factor = (2 * m_electron * e_charge) / (hbar ** 2)

# Functii pentru calculul matricilor

def get_k(i, E):
    # Folosim complex128 deoarece sub bariera k devine imaginar
    return np.sqrt(t_factor * (E - V[i]) + 0j)

def get_r(i, E):
    return np.sqrt((E - V[i]) / (E - V[i-1]) + 0j)

def get_M_transfer(i, E):
    r = get_r(i, E)
    M = 0.5 * np.array([
        [1 + r, 1 - r],
        [1 - r, 1 + r]
    ], dtype=complex)
    return M

def get_M_transport(i, E):
    k = get_k(i, E)
    d = widths_m[i-1] # i-1 pentru ca elementul 0 din widths corespunde domeniului 1
    M = np.array([
        [np.exp(-1j * k * d), 0],
        [0, np.exp(1j * k * d)]
    ], dtype=complex)
    return M

def calculate_transmittance(E):
    n = len(V) - 1 # Numarul de interfete
    
    # Initializam matricea totala ca matrice identitate
    M_total = np.identity(2, dtype=complex)
    
    # Inmultim succesiv matricile conform algoritmului:
    # M = M_transfer(1) * M_transport(1) * M_transfer(2) * ... * M_transfer(n)
    for i in range(1, n):
        M_total = M_total @ get_M_transfer(i, E)
        M_total = M_total @ get_M_transport(i, E)
    
    # Inmultirea cu ultima matrice de transfer (la ultima interfata)
    M_total = M_total @ get_M_transfer(n, E)
    
    # Transmitanta este 1 / |M_00|^2
    return 1.0 / (np.abs(M_total[0, 0]) ** 2)

# Calcul numeric si Verificari

# 1. Verificare pentru energia de la seminar (E0 = 0.0602471 eV)
E0 = 0.0602471
T_E0 = calculate_transmittance(E0)
print(f"La energia E0 = {E0} eV:")
print(f" -> Transmitanta calculata: {T_E0:.8e}")
print(f" -> Valoarea asteptata (MathCAD): 1.82635638e-01\n")

# 2. Generarea spectrului de transmitanta pentru grafic
E_points = np.linspace(0.000001, 0.3, 30000)
T_points = np.array([calculate_transmittance(E) for E in E_points])

# Reprezentare Grafica
plt.figure(figsize=(10, 5))
plt.plot(E_points, T_points, color='red', lw=1.5, label='$RR_j$ (Transmitanta)')

# Setari axe identice cu cele din MathCAD
plt.yscale('log')
plt.xlim(0, 0.3)
plt.ylim(1e-25, 1)

# Adaugam linii verticale ghid pentru a observa spike-urile de rezonanta
plt.xticks(np.arange(0, 0.31, 0.02))
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.xlabel('$E_j$ (eV)', fontsize=12)
plt.ylabel('$RR_j$', fontsize=12)
plt.title('Spectrul de Transmitanta prin Gropi si Bariere Finite', fontsize=14)
plt.legend()
plt.show()