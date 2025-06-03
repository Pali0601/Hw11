import numpy as np
from scipy.linalg import solve

a, b = 0, 1
alpha, beta = 1, 2
h = 0.1
n = int((b - a) / h)
x_fd = np.linspace(a, b, n + 1)

def p(x): return 1 + x**2
def q(x): return 2 * x
def r(x): return -(1 + x**2) * (1 - np.exp(-x))

A = np.zeros((n - 1, n - 1))
F = np.zeros(n - 1)

for i in range(1, n):
    xi = x_fd[i]
    pi = p(xi)
    qi = q(xi)
    ri = r(xi)

    a_i = 1 - (h / 2) * pi
    b_i = 2 + h**2 * qi
    c_i = 1 + (h / 2) * pi
    d_i = -h**2 * ri

    if i != 1:
        A[i-1, i-2] = a_i
    A[i-1, i-1] = b_i
    if i != n - 1:
        A[i-1, i] = c_i

    F[i-1] = d_i

F[0] += (1 - (h / 2) * p(x_fd[1])) * alpha
F[-1] += (1 + (h / 2) * p(x_fd[n - 1])) * beta

Y_inner = solve(A, F)

Y_fd = np.zeros(n + 1)
Y_fd[0] = alpha
Y_fd[1:n] = Y_inner
Y_fd[n] = beta

for x, y in zip(x_fd, Y_fd):
    print(f"x = {x:.2f}, y = {y:.6f}")
