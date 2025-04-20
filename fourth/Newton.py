import numpy as np

def F(x, y):
    """
    Система уравнений:
    F1(x, y) = sin(x - 1) + y - 1.3
    F2(x, y) = x - sin(y + 1) - 0.8
    """
    f1 = 2*(x**3) - y**2 - 1
    f2 = x * (y**3) - y - 4
    return np.array([f1, f2])


def J(x, y):
    """
    Якобиан для системы:
    f1 = 2*(x**3) - y**2 - 1
    f2 = x * (y**3) - y - 4
    """
    df1_dx = 6 * x ** 2
    df1_dy = -2 * y

    df2_dx = y ** 3
    df2_dy = 3 * x * y ** 2 - 1

    return np.array([
        [df1_dx, df1_dy],
        [df2_dx, df2_dy]
    ])

def delta_manual(x, y):
    f1, f2 = -F(x, y)
    a, b = J(x, y)[0]
    c, d = J(x, y)[1]
    if abs(a) < 1e-12:
        raise ValueError("Значение a слишком близко к нулю")
    denominator = d - (c * b) / a
    if abs(denominator) < 1e-12:
        raise ValueError("Знаменатель близок к нулю")
    delta_y = (f2 - (c / a) * f1) / denominator
    delta_x = (f1 - b * delta_y) / a
    return np.array([delta_x, delta_y])

def newton_manual_full(x0, y0, tol=1e-4, max_iter=50):
    x, y = x0, y0
    for i in range(max_iter):
        delta = delta_manual(x, y)
        norm = np.sqrt(delta[0]**2 + delta[1]**2)
        x_new = x + delta[0]
        y_new = y + delta[1]
        print(f"Итерация {i+1}: x = {x_new:.10f}, y = {y_new:.10f}, Δ = {delta}, ||Δ|| = {norm:.6f}")
        if norm < tol:
            print("Метод остановлен по критерию точности.")
            return x_new, y_new
        x, y = x_new, y_new
    print("Лимит итераций достигнут")
    return x, y

x0, y0 = 1.8200, 0.5000
x_sol, y_sol = newton_manual_full(x0, y0)
print("Найденное решение:")
print(f"x ≈ {x_sol:.15f}")
print(f"y ≈ {y_sol:.15f}")
