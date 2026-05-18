import numpy as np
import matplotlib.pyplot as plt

# Parameterii
xmax = 1  # Lungimea domeniului spatial
tmax = 1  # Timpul total de simulare
n = 10  # Numarul de spatii (n+1 puncte in total)
m = 100  # Numarul de pasi in timp (m+1 puncte in total)
D = 2.2697 * (10**-2)  # Coeficientul de difuzie

# Parametri pentru discretizare
x = np.linspace(0, xmax, n + 1)  # Puncte in spatiu
dx = x[1] - x[0]  # Marimea pasului in spatiu
t = np.linspace(0, tmax, m + 1)  # Puncte in timp
dt = t[1] - t[0]  # Marimea pasului in timp

r = D * dt / (2 * dx**2) 

c = np.zeros((m + 1, n + 1))

c[0, :] = 0  # Conditie initiala (u(x, 0) = 0 pt oricare x x)
c[:, 0] = 4.376  # conditii stanga (u(0, t) = 4.376)
c[:, -1] = 0  # conditii dreapta (u(1, t) = 0)

A = np.zeros((n - 1, n - 1))
B = np.zeros((n - 1, n - 1))

for i in range(n - 1):
    if i > 0:
        A[i, i - 1] = -r
        B[i, i - 1] = r
    A[i, i] = 1 + 2 * r
    B[i, i] = 1 - 2 * r
    if i < n - 2:
        A[i, i + 1] = -r
        B[i, i + 1] = r

for j in range(0, m):
    # Computa partea dreapta
    b = B @ c[j, 1:-1]
    # Rezolva pt Ax = b
    c[j + 1, 1:-1] = np.linalg.solve(A, b)

for j in range(0, m + 1, 100):
    plt.plot(x, c[j, :], label=f't = {t[j]:.2f}')

plt.xlabel('x')
plt.ylabel('C')
plt.title('Crank-Nicholson Solution: C(t, x)')
plt.legend()
plt.grid(True, linestyle='--', color='g', alpha=0.15, which='both')
plt.minorticks_on()
plt.show()