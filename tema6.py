import numpy as np
import matplotlib.pyplot as plt

# 1. CONSTANTE FIZICE & PARAMETRI SISTEM
h_plank = 6.626026e-34       # Constanta lui Planck
me = 9.1e-31                 # Masa electronului
ee = 1.6e-19                 # Sarcina electronului
hbar = h_plank / (2 * np.pi) # h barat

L = 10.0                     # Latimea gropii (nm)
V0 = 0.1                     # Inaltimea barierei (eV)

# PARTEA 1: METODA TRADITIONALAA + NEWTON-RAPHSON (Stari Pare si Impare)

def get_k(E):
    # Vectorul de unda k in nm^-1
    return (np.sqrt(2 * me * E * ee) / hbar) * 10**-9

def get_alpha(E):
    # Constanta de atenuare alpha in nm^-1
    return (np.sqrt(2 * me * (V0 - E) * ee) / hbar) * 10**-9

# Functiile transcendente f(E) ale caror radacini le cautam
def f_par(E):
    return get_k(E) * np.tan(get_k(E) * L / 2) - get_alpha(E)

def f_impar(E):
    # Folosim 1/tan pentru cotangenta
    return get_k(E) * (1.0 / np.tan(get_k(E) * L / 2)) + get_alpha(E)

def newton_raphson(f, E_start, epsilon=1e-6, max_iter=100):
    E = E_start
    for _ in range(max_iter):
        f_val = f(E)
        # Derivata numerica dF/dE
        dE = 1e-8
        f_prime = (f(E + dE) - f_val) / dE
        
        E_next = E - f_val / f_prime
        if abs(E_next - E) < epsilon:
            return E_next
        E = E_next
    return E

# PARTEA 2: METODA TIRULUI (SHOOTING METHOD)

def potential(x):
    # Centram groapa intre x = 10nm si x = 20nm pentru a simula un domeniu extins
    if x < 10.0:
        return V0
    elif 10.0 <= x <= 20.0:
        return 0.0
    else:
        return V0

def shooting_method(E, dx=0.01, x_max=50.0):
    steps = int(x_max / dx)
    
    # Factorul adimensional t din MathCAD adaptat pentru pasul dx ales
    t_factor = (2 * (dx**2) * 10**-18 * ee * me) / (hbar**2)
    
    # Conditiile initiale
    psi_prev2 = 0.0  # psi_1
    psi_prev1 = 1.0  # psi_2
    
    for i in range(3, steps + 1):
        x = i * dx
        V_x = potential(x)
        # Ecuatia Schrodinger cu diferente finite
        psi_curr = (t_factor * (V_x - E) + 2) * psi_prev1 - psi_prev2
        
        psi_prev2 = psi_prev1
        psi_prev1 = psi_curr
        
        # Guardrail impotriva overflow-ului numeric in regiunile interzise clasic
        if abs(psi_curr) > 1e30:
            return np.sign(psi_curr) * 1e30
            
    return psi_prev1

def find_shooting_energies(dx=0.01, pasE=0.0002):
    E_array = np.arange(pasE, V0, pasE)
    psi_endpoints = []
    
    for E in E_array:
        psi_endpoints.append(shooting_method(E, dx=dx))
        
    psi_endpoints = np.array(psi_endpoints)
    
    # Identificam intersectiile cu axa zero (schimbarile de semn)
    solutions = []
    for i in range(1, len(psi_endpoints)):
        if psi_endpoints[i-1] * psi_endpoints[i] < 0:
            # Interpoleaza liniar mijlocul intervalului energiei
            E_sol = (E_array[i-1] + E_array[i]) / 2.0
            solutions.append(E_sol)
            
    return solutions

# EXECUTIE SI DETECTARE AUTOMATA A STARILOR

print("--- RULARE METODA TRADITIONALA (NEWTON-RAPHSON) ---")
# Folosim aproximari initiale din graficul transcendental din seminar
E_stari_pare_init = [0.01]   # n=1,3,5...
E_stari_impare_init = [0.03, 0.07] # n=2,4,6...

E_traditional = {}

# Calcul stari pare
for E_start in E_stari_pare_init:
    res = newton_raphson(f_par, E_start)
    # Mapam riguros pe numerele cuantice corespunzatoare
    if res < 0.02: E_traditional[1] = res
    elif res < 0.08: E_traditional[3] = res
    else: E_traditional[5] = res

# Calcul stari impare
for E_start in E_stari_impare_init:
    res = newton_raphson(f_impar, E_start)
    if res < 0.035: E_traditional[2] = res
    elif res < 0.085: E_traditional[4] = res
    else: E_traditional[6] = res

# Afisare rezultate exacte Newton-Raphson
for n in sorted(E_traditional.keys()):
    print(f"Starea n={n}: E = {E_traditional[n]:.6f} eV")

print("\n--- RULARE METODA TIRULUI (SHOOTING METHOD) ---")
print("Se calculeaza pentru pasi diferiti pe axa X (convergenta)...")
E_tir_05 = find_shooting_energies(dx=0.05)
E_tir_01 = find_shooting_energies(dx=0.01)

# REZULTATUL FINAL: COMPLETAREA TABELULUI
print("\n" + "="*65)
print("TABEL COMPARATIV REZULTATE FINALE (CONVERGENTA SI PRECIZIE)")
print("="*65)
print(f"{'n':<4} | {'E_traditional (eV)':<18} | {'E_tir (dx=0.05nm)':<18} | {'E_tir (dx=0.01nm)':<18}")
print("-"*65)

# Sortam starile de la n=1 la n=6
for idx, n in enumerate(range(1, 7)):
    E_trad_val = E_traditional.get(n, 0.0)
    
    # Extragerea valorilor corespunzatoare din vectorii metodei tirului
    # (scoate solutiile in ordine crescatoare a energiei)
    E_tir_05_val = E_tir_05[idx] if idx < len(E_tir_05) else float('nan')
    E_tir_01_val = E_tir_01[idx] if idx < len(E_tir_01) else float('nan')
    
    print(f"{n:<4} | {E_trad_val:<18.6f} | {E_tir_05_val:<18.4f} | {E_tir_01_val:<18.4f}")
print("="*65)