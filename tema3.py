import numpy as np
import matplotlib.pyplot as plt

#Constante(GaAs) 
hbar = 1.0545718e-34
m_eff = 0.067 * 9.1093837e-31  # Masa efectiva GaAs
eV = 1.60217663e-19

L = 10.0e-9  # Latura firului: 10 nm

# Functie pentru energia analitia a gropii infinite 1D
def energy_1d(n):
    return (hbar**2 * np.pi**2 * n**2) / (2 * m_eff * L**2)

# 1. Calculul subbenzilor energetice Ey,z = Ey + Ez
states = {
    "1,1": (1, 1),
    "1,2 / 2,1": (1, 2), # Degenerate
    "2,2": (2, 2)
}

print("Stare (ny, nz) | Energie totala (meV)")
print("-" * 35)
for label, (ny, nz) in states.items():
    E_total = (energy_1d(ny) + energy_1d(nz)) / eV * 1000
    print(f"     {label}     | {E_total:.2f} meV")

# 2. Generarea densitatilor de probabilitate
grid_points = 200
y = np.linspace(0, L, grid_points)
z = np.linspace(0, L, grid_points)
Y, Z = np.meshgrid(y, z)

def wave_function_2d(ny, nz, Y, Z):
    psi_y = np.sqrt(2/L) * np.sin(ny * np.pi * Y / L)
    psi_z = np.sqrt(2/L) * np.sin(nz * np.pi * Z / L)
    return psi_y * psi_z

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
pairs = [(1,1), (1,2), (2,1), (2,2)]
labels = ['(a) n_y=1, n_z=1', '(b) n_y=1, n_z=2', '(c) n_y=2, n_z=1', '(d) n_y=2, n_z=2']

for idx, (ny, nz) in enumerate(pairs):
    ax = axs[idx//2, idx%2]
    psi = wave_function_2d(ny, nz, Y, Z)
    prob_density = psi**2
    
    contour = ax.contourf(Y*1e10, Z*1e10, prob_density, levels=50, cmap='jet')
    ax.set_title(labels[idx])
    ax.set_xlabel('z ($\AA$)')
    ax.set_ylabel('y ($\AA$)')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()