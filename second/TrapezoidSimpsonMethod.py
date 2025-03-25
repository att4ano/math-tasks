import numpy as np
from scipy.integrate import quad
from sympy import symbols, cos, diff, lambdify, init_printing

init_printing()
x = symbols('x')
u = x + x**3
f_sym = cos(u)

f2_sym = diff(f_sym, x, 2)
f4_sym = diff(f_sym, x, 4)

f2 = lambdify(x, f2_sym, 'numpy')
f4 = lambdify(x, f4_sym, 'numpy')
f = lambdify(x, f_sym, 'numpy')

x_vals = np.linspace(0, 1, 10_000)
max_f2 = np.max(np.abs(f2(x_vals)))
max_f4 = np.max(np.abs(f4(x_vals)))

print(f"Оценка max |f''(x)| на [0,1]: {max_f2:.4f}")
print(f"Оценка max |f⁽⁴⁾(x)| на [0,1]: {max_f4:.4f}")

a, b = 0, 1
eps = 1e-3

N_trap = int(np.ceil(np.sqrt((max_f2 * (b - a)**3) / (12 * eps))))
N_simp = int(np.ceil(((max_f4 * (b - a)**5) / (180 * eps))**(1/4)))
if N_simp % 2 != 0:
    N_simp += 1

def trapezoidal(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def simpson(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h / 3 * (y[0] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N-1:2]) + y[N])

I_trap = trapezoidal(f, a, b, N_trap)
I_simp = simpson(f, a, b, N_simp)
I_exact, _ = quad(f, a, b)

print("\n=== Расчёт N ===")
print(f"N для трапеций: {N_trap}")
print(f"N для Симпсона: {N_simp}")

print("\n=== Результаты интегрирования ===")
print(f"[Трапеции]   I ≈ {I_trap:.6f}, погрешность ≈ {abs(I_trap - I_exact):.2e}")
print(f"[Симпсон]    I ≈ {I_simp:.6f}, погрешность ≈ {abs(I_simp - I_exact):.2e}")
print(f"[quad]       I ≈ {I_exact:.6f} (почти точное значение)")