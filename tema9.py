import numpy as np
import matplotlib.pyplot as plt

# 1. CONSTANTE FIZICE & PARAMETRI SISTEM
h_plank = 6.625e-34          # Constanta lui Planck
hbar = h_plank / (2 * np.pi) # h-bar
m0 = 9.1e-31                 # Masa la repaus a electronului
e_charge = 1.6e-19           # Sarcina electronului
meff = 0.063 * m0            # Masa efectiva in GaAs

dx = 1e-10                   # Pasul de discretizare (1 Angstrom)
V0 = 0.2                     # Inaltimea maxima a potentialului (eV)
n_cells = 10                 # Numarul de celule elementare

# O celula elementara are 100 de pasi (de la 0 la 100*dx = 10 nm)
steps_per_cell = 100 
N_spatial = n_cells * steps_per_cell  # Numarul total de puncte (1000 pasi)

# Factorul constant pentru vectorul de unda k
k_constant = (2 * meff * e_charge) / (hbar ** 2)

# 2. DEFINIREA POTENTIALULUI TRIUNGHIULAR
def get_triangular_potential():
    V = np.zeros(N_spatial + 1)
    for i in range(N_spatial + 1):
        # Aflam pozitia relativa in interiorul celulei elementare curente
        aux = i % steps_per_cell
        
        # Replicam logica conditionala de panta din MathCAD
        if aux > 50:
            aux_val = 100 - aux
        else:
            aux_val = aux
            
        V[i] = V0 * (2.0 * aux_val / steps_per_cell)
    return V

V_profile = get_triangular_potential()

# 3. ALGORITMUL MATRICILOR DE TRANSFER & TRANSPORT
def get_k_vector(E):
    # Adaugam +0j pentru a asigura suportul numerelor complexe in regiunile barierei
    return np.sqrt(k_constant * (E - V_profile) + 0j)

def calculate_system(E):
    k = get_k_vector(E)
    
    # CALCUL PENTRU O CELULA ELEMENTARA (MCE) 
    M_CE = np.identity(2, dtype=complex)
    for i in range(1, steps_per_cell):
        R = k[i] / k[i-1]
        M_transfer = 0.5 * np.array([[1+R, 1-R], [1-R, 1+R]], dtype=complex)
        M_transport = np.array([[np.exp(-1j*k[i]*dx), 0], [0, np.exp(1j*k[i]*dx)]], dtype=complex)
        M_CE = M_CE @ M_transfer @ M_transport
        
    # Ultima interfata a celulei elementare
    R_last_ce = k[steps_per_cell] / k[steps_per_cell - 1]
    M_transfer_last_ce = 0.5 * np.array([[1+R_last_ce, 1-R_last_ce], [1-R_last_ce, 1+R_last_ce]], dtype=complex)
    M_CE = M_CE @ M_transfer_last_ce
    
    # Valoarea Kramers: jumatate din modulul urmei
    TR_Kramers = 0.5 * np.abs(np.trace(M_CE))
    
    # CALCUL STRUCTURA TOTALA (Pentru Transmitanta T)
    M_total = np.identity(2, dtype=complex)
    for i in range(1, N_spatial):
        R = k[i] / k[i-1]
        M_transfer = 0.5 * np.array([[1+R, 1-R], [1-R, 1+R]], dtype=complex)
        M_transport = np.array([[np.exp(-1j*k[i]*dx), 0], [0, np.exp(1j*k[i]*dx)]], dtype=complex)
        M_total = M_total @ M_transfer @ M_transport
        
    # Ultima interfata a intregului sistem
    R_end = k[N_spatial] / k[N_spatial - 1]
    M_transfer_end = 0.5 * np.array([[1+R_end, 1-R_end], [1-R_end, 1+R_end]], dtype=complex)
    M_total = M_total @ M_transfer_end
    
    # Formula transmitantei
    T = (k[N_spatial].real / k[0].real) * (1.0 / (np.abs(M_total[0, 0]) ** 2))
    
    return T, TR_Kramers

# 4. SCANAREA ENERGIEI & REPREZENTAREA GRAFICA
# # Generam spectrul de energii de la 0 la 0.5 eV (1000 de puncte)
E_points = np.linspace(0.000001, 0.5, 1000)

T_results = []
Kramers_results = []

for E in E_points:
    T, TR = calculate_system(E)
    T_results.append(T)
    Kramers_results.append(TR)

T_results = np.array(T_results)
Kramers_results = np.array(Kramers_results)

# GENERARE GRAFICE 
plt.figure(figsize=(12, 5))

# Graficul 1: Spectrul complet de Transmitanta in scara logaritmica
plt.subplot(1, 2, 1)
plt.plot(E_points, T_results, color='red', lw=1.5, label='T(E)')
plt.axhline(1.0, color='blue', linestyle=':', alpha=0.7, label='1.0')
plt.yscale('log')
plt.xlim(0, 0.5)
plt.ylim(1e-7, 10)
plt.xlabel('E_j (eV)')
plt.ylabel('out_j (Transmitanta)')
plt.title('Spectrul complet al Transmitantei T(E)')
plt.grid(True, which="both", alpha=0.3)
plt.legend()

# Graficul 2: Detaliu Kramers vs Transmitanta
plt.subplot(1, 2, 2)
plt.plot(E_points, T_results, color='red', lw=1.5, label='out_j (T)')
plt.plot(E_points, Kramers_results, color='blue', lw=1.5, label='TR(E) - Kramers')
plt.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Limita Banda = 1')

plt.xlim(0.08, 0.15)
plt.ylim(-1, 2)
plt.xlabel('E_j (eV)')
plt.title('Verificarea Conditiei Kramers vs Banda Permisa')
plt.grid(True, alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()