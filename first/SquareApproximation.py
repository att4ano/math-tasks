import numpy as np
import matplotlib.pyplot as plt

a = 0.617
b = -0.54
f = lambda x: np.exp(a * x**3 + b * x)

x_vals = np.linspace(-1.0, 1.0, 6)
y_vals = f(x_vals)

x_sample = np.array([-0.6, 0.0, 0.6])
y_sample = f(x_sample)

A = np.vstack([np.ones_like(x_sample), x_sample, x_sample**2]).T
coeffs_exact = np.linalg.solve(A, y_sample)

a0_e, a1_e, a2_e = coeffs_exact
print("Парабола (точечное приближение):")
print(f"P_exact(x) = {a0_e:.6f} + ({a1_e:.6f})*x + ({a2_e:.6f})*x^2")

A_lsq = np.vstack([np.ones_like(x_vals), x_vals, x_vals**2]).T
coeffs_lsq = np.linalg.lstsq(A_lsq, y_vals, rcond=None)[0]

a0_l, a1_l, a2_l = coeffs_lsq
print("\nПарабола (среднеквадратичное приближение):")
print(f"P_lsq(x) = {a0_l:.6f} + ({a1_l:.6f})*x + ({a2_l:.6f})*x^2")

x_plot = np.linspace(-1.0, 1.0, 200)
f_plot = f(x_plot)

P_exact = a0_e + a1_e*x_plot + a2_e*x_plot**2
P_lsq = a0_l + a1_l*x_plot + a2_l*x_plot**2

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f_plot, label='f(x)', linewidth=2)
plt.plot(x_plot, P_exact, '--', label='Точечная парабола (3 точки)')
plt.plot(x_plot, P_lsq, '-.', label='МНК парабола (6 точек)')
plt.scatter(x_vals, y_vals, color='black', zorder=5)
plt.title('Параболическое приближение функции')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

x_control = 0.8
true_val = f(x_control)
exact_val = a0_e + a1_e*x_control + a2_e*x_control**2
lsq_val = a0_l + a1_l*x_control + a2_l*x_control**2

print(f"\nКонтрольная точка x = 0.8:")
print(f"f(0.8) = {true_val:.6f}")
print(f"P_exact(0.8) = {exact_val:.6f}")
print(f"P_lsq(0.8) = {lsq_val:.6f}")