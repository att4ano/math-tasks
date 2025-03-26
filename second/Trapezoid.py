import numpy as np
from scipy.integrate import quad
from sympy import symbols, cos, diff, lambdify, init_printing

init_printing()
x = symbols('x')

f_sym = cos(x + x**3)

f2_sym = diff(f_sym, x, 2)
print(f2_sym)
f = lambdify(x, f_sym, 'numpy')
f2 = lambdify(x, f2_sym, 'numpy')

x_values = np.linspace(0, 1, 10000)
f2_values = f2(x_values)  # |f''(x)|

M2 = np.max(f2_values)

print(f"Максимальное значение второй производной:", M2)

eps = 1e-3
a, b = 0, 1

N_trap = int(np.ceil(np.sqrt((M2 * (b - a)**3) / (12 * eps))))
print(f"Расчетное N для трапеций: {N_trap}")

# Метод трапеций
def trapezoidal(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

I_exact, _ = quad(f, a, b)
I_trap = trapezoidal(f, a, b, N_trap)
print("\n=== Результаты ===")
print(f"Метод трапеций (n={N_trap}): {I_trap:.6f}")
print(f"Точное значение: {I_exact:.6f}")
print(f"Фактическая погрешность: {abs(I_trap - I_exact):.2e}")