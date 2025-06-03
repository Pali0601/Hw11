import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

def ode_system(x, Y):
    y, yp = Y
    f = ((2 * x * y) - 2 * (1 + x**2) * yp - (1 + x**2) * (1 - np.exp(-x))) / (1 + x**2)
    return [yp, f]

def shooting_residual(s):
    sol = solve_ivp(ode_system, [0, 1], [1, s], t_eval=[1])
    return sol.y[0][-1] - 2

sol_shoot = root_scalar(shooting_residual, bracket=[0, 10], method='brentq')
s_correct = sol_shoot.root

x_vals = np.linspace(0, 1, 11)
sol_ivp = solve_ivp(ode_system, [0, 1], [1, s_correct], t_eval=x_vals)

x_shoot, y_shoot = sol_ivp.t, sol_ivp.y[0]

for x, y in zip(x_shoot, y_shoot):
    print(f"x = {x:.2f}, y = {y:.6f}")
