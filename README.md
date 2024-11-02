# transcedental_equations
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return -x**2 + 3

# Bisection method implementation
def bisection(f, a, b, tol=1e-6):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        if f(midpoint) == 0:
            return midpoint  # Exact solution found
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2

# Solve using Bisection
a, b = 0, 2
root_bisection = bisection(f, a, b)
print(f"Root (Bisection method): x = {root_bisection}")

# Plotting
x_vals = np.linspace(-2, 2, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x) = -x^2 + 3$', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(root_bisection, color='red', linestyle='--', label=f'Bisection Root at x ≈ {root_bisection:.3f}')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Define the function and its derivative
def f(x):
    return -x**2 + 3

def df(x):
    return -2 * x

# Solve using Newton-Raphson
x0 = 1.5
root_newton = newton(f, x0, fprime=df)
print(f"Root (Newton-Raphson method): x = {root_newton}")

# Plotting
x_vals = np.linspace(-2, 2, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x) = -x^2 + 3$', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(root_newton, color='green', linestyle='--', label=f'Newton-Raphson Root at x ≈ {root_newton:.3f}')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Newton-Raphson Method Solution for $-x^2 + 3 = 0$")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return -x**2 + 3

# Secant method implementation
def secant(f, x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        if abs(f(x1)) < tol:
            return x1
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
    return x1

# Solve using Secant method
x0_sec, x1_sec = 0, 2
root_secant = secant(f, x0_sec, x1_sec)
print(f"Root (Secant method): x = {root_secant}")

# Plotting
x_vals = np.linspace(-2, 2, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x) = -x^2 + 3$', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(root_secant, color='purple', linestyle='--', label=f'Secant Root at x ≈ {root_secant:.3f}')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Secant Method Solution for $-x^2 + 3 = 0$")
plt.legend()
plt.grid(True)
plt.show()

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Bisection Method Solution for $-x^2 + 3 = 0$")
plt.legend()
plt.grid(True)
plt.show()
