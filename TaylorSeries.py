import sympy as sp
import matplotlib.pyplot as plt


def taylor_series(func, x, point, order, flag=True):
     series_result = sp.series(func, x + 1, point, order)
     if not flag:
          series_result = series_result.removeO()

     return str(series_result)

while True:
     user_input = input("Введите функцию (используйте 'x' как переменную): ")
     try:
          func = sp.sympify(user_input)
          break
     except sp.SympifyError:
          print("Ошибка: введённое выражение некорректно. Попробуйте снова.")

x = sp.symbols('x')
point = float(input("Введите точку, вокруг которой вычисляется ряд (например, 0): "))
order = int(input("Введите порядок ряда (например, 6): "))
flag = input("Оставить остаточный член? (y/n): ").strip().lower() == 'y'

result = taylor_series(func, x, point, order, flag)
print("\nРезультат:")
print(result)
