import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Определение функции дифференциального уравнения
def f(x, y):
    return x * y + 0.1 * y ** 2


# Начальные условия
x0 = 0
y0 = 0.4
x_end = 1


# 1. Метод Эйлера
def euler_method(h):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])

    return x, y


# 2. Метод Рунге-Кутты 4-го порядка
def runge_kutta_method(h):
    x = np.arange(x0, x_end + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        k1 = f(x[i - 1], y[i - 1])
        k2 = f(x[i - 1] + h / 2, y[i - 1] + h * k1 / 2)
        k3 = f(x[i - 1] + h / 2, y[i - 1] + h * k2 / 2)
        k4 = f(x[i - 1] + h, y[i - 1] + h * k3)

        y[i] = y[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y


# 3. Метод степенных рядов (n=4)
def power_series_method():
    # Вычисляем коэффициенты разложения
    a0 = y0
    a1 = f(x0, a0)  # y'(0) = f(0, y(0))

    # y'' = y + xy' + 0.2yy'
    a2 = (a0 + x0 * a1 + 0.2 * a0 * a1) / 2

    # y''' = 2y' + xy'' + 0.2(y'^2 + yy'')
    a3 = (2 * a1 + x0 * a2 + 0.2 * (a1 ** 2 + a0 * a2)) / 6

    # y'''' = 3y'' + xy''' + 0.2(3y'y'' + yy''')
    a4 = (3 * a2 + x0 * a3 + 0.2 * (3 * a1 * a2 + a0 * a3)) / 24

    # Функция-полином
    def y_approx(x):
        return a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4

    x = np.linspace(x0, x_end, 100)
    y = y_approx(x)

    return x, y, [a4, a3, a2, a1, a0]


# 4. Метод последовательных приближений (Пикара, n=3)
def picard_method():
    # Первое приближение y1(x) = y0 + ∫(0.1*y0²)dt от 0 до x
    y1 = lambda x: y0 + 0.1 * y0 ** 2 * x

    # Второе приближение y2(x) = y0 + ∫(t*y1 + 0.1*y1²)dt
    def y2(x):
        return y0 + 0.1 * y0 ** 2 * x + (0.1 * y0 ** 3 / 3 + 0.01 * y0 ** 4 / 2) * x ** 2

    # Третье приближение (упрощенное)
    def y3(x):
        return y0 + 0.1 * y0 ** 2 * x + (0.1 * y0 ** 3 / 3 + 0.01 * y0 ** 4 / 2) * x ** 2 + \
            (0.1 * y0 ** 4 / 12 + 0.01 * y0 ** 5 / 6 + 0.001 * y0 ** 6 / 9) * x ** 3

    x = np.linspace(x0, x_end, 100)
    return x, y3(x)


# Вычисление всех решений
x_euler1, y_euler1 = euler_method(0.1)
x_euler2, y_euler2 = euler_method(0.05)
x_rk1, y_rk1 = runge_kutta_method(0.1)
x_rk2, y_rk2 = runge_kutta_method(0.05)
x_ps, y_ps, coeffs = power_series_method()
x_picard, y_picard = picard_method()

# Распаковываем коэффициенты степенного ряда
a4, a3, a2, a1, a0 = coeffs

# Оценка погрешностей (сравнение с Рунге-Кутта h=0.05 как эталоном)
rk_ref = interp1d(x_rk2, y_rk2, kind='cubic', fill_value="extrapolate")


def calculate_error(x, y, method_name):
    try:
        y_ref = rk_ref(x)
        error = np.max(np.abs(y - y_ref))
        print(f"Максимальная погрешность метода {method_name}: {error:.6f}")
    except:
        print(f"Не удалось оценить погрешность для {method_name}")


calculate_error(x_euler1, y_euler1, "Эйлера (h=0.1)")
calculate_error(x_euler2, y_euler2, "Эйлера (h=0.05)")
calculate_error(x_rk1, y_rk1, "Рунге-Кутта (h=0.1)")
calculate_error(x_ps, y_ps, "степенных рядов")
calculate_error(x_picard, y_picard, "последовательных приближений")

# Построение графиков
plt.figure(figsize=(12, 8))
plt.plot(x_euler1, y_euler1, 'o-', markersize=4, label='Метод Эйлера (h=0.1)')
plt.plot(x_euler2, y_euler2, 'o-', markersize=4, label='Метод Эйлера (h=0.05)')
plt.plot(x_rk1, y_rk1, 's-', markersize=4, label='Метод Рунге-Кутта (h=0.1)')
plt.plot(x_rk2, y_rk2, 's-', markersize=4, label='Метод Рунге-Кутта (h=0.05)')
plt.plot(x_ps, y_ps, '-', linewidth=2, label='Метод степенных рядов (n=4)')
plt.plot(x_picard, y_picard, '-', linewidth=2, label='Метод последовательных приближений (n=3)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.title('Сравнение методов решения задачи Коши', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Вывод таблицы значений
print("\nТаблица значений:")
print("x\tЭйлер(h=0.1)\tЭйлер(h=0.05)\tРунге-Кутта(h=0.1)\tРунге-Кутта(h=0.05)\tСтеп.ряды\tПосл.прибл.")
for i in range(len(x_euler1)):
    x = x_euler1[i]

    # Значения Эйлера
    euler1_val = y_euler1[i]
    euler2_val = y_euler2[i * 2] if i * 2 < len(y_euler2) else np.nan

    # Значения Рунге-Кутта
    rk1_val = y_rk1[i]
    rk2_val = y_rk2[i * 2] if i * 2 < len(y_rk2) else np.nan

    # Значение степенного ряда
    ps_val = np.polyval([a4, a3, a2, a1, a0], x)

    # Значение метода Пикара (ближайшее)
    picard_idx = np.argmin(np.abs(x_picard - x))
    picard_val = y_picard[picard_idx]

    print(f"{x:.1f}\t{euler1_val:.6f}\t{euler2_val:.6f}\t{rk1_val:.6f}\t{rk2_val:.6f}\t{ps_val:.6f}\t{picard_val:.6f}")