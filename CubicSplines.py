import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

a = 0.617
b = -0.54
f = lambda x: np.exp(a * x**3 + b * x)

x_nodes = np.linspace(-1.0, 1.0, 5)
y_nodes = f(x_nodes)

spline = CubicSpline(x_nodes, y_nodes, bc_type='natural')

x_plot = np.linspace(-1.0, 1.0, 300)
f_plot = f(x_plot)
s_plot = spline(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f_plot, label='f(x)', linewidth=2)
plt.plot(x_plot, s_plot, '--', label='Кубический сплайн (4 промежутка)')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Узлы сплайна')
plt.title('Аппроксимация функции кубическим сплайном')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

x_control = 0.8
true_val = f(x_control)
spline_val = spline(x_control)

print(f"Контрольная точка x = 0.8:")
print(f"f(0.8) = {true_val:.6f}")
print(f"S(0.8) = {spline_val:.6f}")