import sympy as sp

x = sp.symbols('x')
def lagrange_polynomial(x_values, y_values):
    n = len(x_values)
    poly = 0

    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        poly += term

    return sp.simplify(poly)

def input_nodes():
    x_values = list(map(float, input("Введите узлы интерполяции через пробел: ").split()))
    return x_values

def input_function():
    while True:
        user_input = input("Введите функцию (используйте 'x' как переменную): ")
        try:
            function = sp.sympify(user_input)
            return function
        except sp.SympifyError:
            print("Ошибка: введённое выражение некорректно. Попробуйте снова.")

if __name__ == "__main__":
    x_values = input_nodes()
    func = input_function()
    y_values = [func.subs(x, xi).evalf() for xi in x_values]
    lagrange_poly = lagrange_polynomial(x_values, y_values)
    print("Интерполяционный многочлен Лагранжа:")
    print(lagrange_poly)