import numpy as np


def F(x, y):
    """
    Система уравнений:
    F1(x, y) = 2*(x**3) - y**2 - 1
    F2(x, y) = x * (y**3) - y - 4
    """
    f1 = 2 * (x ** 3) - y ** 2 - 1
    f2 = x * (y ** 3) - y - 4
    return np.array([f1, f2])


def simple_iteration_method(x0, y0, tol=1e-4, max_iter=100, alpha=0.1):
    x, y = x0, y0
    print(f"Начальное приближение: x0 = {x0}, y0 = {y0}")
    print("Итерационный процесс:")

    for i in range(max_iter):
        f1, f2 = F(x, y)
        x_new = x - alpha * f1
        y_new = y - alpha * f2
        delta_x = x_new - x
        delta_y = y_new - y
        norm = np.sqrt(delta_x ** 2 + delta_y ** 2)

        print(
            f"Итерация {i + 1}: x = {x_new:.10f}, y = {y_new:.10f}, Δ = [{delta_x:.6f}, {delta_y:.6f}], ||Δ|| = {norm:.6f}")
        if norm < tol:
            print(f"\nМетод сошелся за {i + 1} итераций")
            return x_new, y_new

        x, y = x_new, y_new

    print("\nДостигнуто максимальное число итераций")
    return x, y

x0, y0 = 1.8, 0.5

tolerance = 1e-3
max_iterations = 100
relaxation_param = 0.01

x_sol, y_sol = simple_iteration_method(x0, y0, tol=tolerance,
                                       max_iter=max_iterations,
                                       alpha=relaxation_param)
print("\nНайденное решение:")
print(f"x ≈ {x_sol:.15f}")
print(f"y ≈ {y_sol:.15f}")
print(f"Значение функций в найденной точке: F(x,y) = {F(x_sol, y_sol)}")