import numpy as np
from scipy.linalg import solve

def p(x): return 1 + x**2
def q(x): return 2 * x
def r(x): return -(1 + x**2) * (1 - np.exp(-x))

N = 9
x_var = np.linspace(0, 1, 100)

phi = lambda i, x: np.sin(i * np.pi * x)
dphi = lambda i, x: i * np.pi * np.cos(i * np.pi * x)

A_var = np.zeros((N, N))
b_var = np.zeros(N)

for i in range(1, N + 1):
    for j in range(1, N + 1):
        integrand = lambda x: p(x) * dphi(i, x) * dphi(j, x) + q(x) * phi(i, x) * phi(j, x)
        A_var[i-1, j-1] = np.trapz(integrand(x_var), x_var)
    integrand_b = lambda x: -r(x) * phi(i, x)
    b_var[i-1] = np.trapz(integrand_b(x_var), x_var)

c = solve(A_var, b_var)

y_var = np.zeros_like(x_var)
for i in range(1, N + 1):
    y_var += c[i-1] * phi(i, x_var)
y_var += (1 - x_var) + x_var

for x, y in zip(x_var, y_var):
    print(f"x = {x:.2f}, y = {y:.6f}")
