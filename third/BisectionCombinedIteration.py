import numpy as np

f = lambda x: 2**x + 5*x - 3
df = lambda x: np.log(2)*2**x + 5

eps = 1e-5
a, b = 0.0, 1.0

def bisection(f, a, b, eps):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Нет гарантии корня на [a, b]")
    iterations = 0
    while (b - a) / 2 > eps:
        c = (a + b) / 2
        fc = f(c)
        print(f"ответ на {iterations} итерации: {c}")
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1
    return (a + b) / 2, iterations

def combined_method(f, df, a, b, eps, max_iter=100):
    x0, x1 = a, b
    for i in range(max_iter):
        secant = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        newton = x1 - f(x1)/df(x1)
        x2 = (secant + newton) / 2
        if abs(x2 - x1) < eps:
            return x2, i+1
        print(f"ответ на {i + 1} итерации: {x2}")
        x0, x1 = x1, x2
    raise RuntimeError("Не сошлось")


phi = lambda x: (3 - 2**x)/5
dphi = lambda x: -(np.log(2) * 2**x)/5

x_test = np.linspace(0, 1, 100)
assert np.max(np.abs(dphi(x_test))) < 1, "Метод итераций может не сойтись"

def simple_iteration(phi, x0, eps, max_iter=1000):
    x_prev = x0
    for i in range(max_iter):
        x_next = phi(x_prev)
        if abs(x_next - x_prev) < eps:
            return x_next, i+1
        print(f"ответ на {i + 1} итерации: {x_next}")
        x_prev = x_next
    raise RuntimeError("Не сошлось")

print("Метод Дихотомии:")
root_bis, it_bis = bisection(f, a, b, eps)

print("Комбинированный метод:")
root_comb, it_comb = combined_method(f, df, a, b, eps)

print("Метод простых итераций:")
root_iter, it_iter = simple_iteration(phi, 0.5, eps)

print(f"[Дихотомия] Корень: {root_bis:.6f}, итераций: {it_bis}")
print(f"[Комбинированный] Корень: {root_comb:.6f}, итераций: {it_comb}")
print(f"[Итерации] Корень: {root_iter:.6f}, итераций: {it_iter}")