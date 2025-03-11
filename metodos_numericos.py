import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, lambdify, sympify

def bisection_method(func, a, b, tol=0.001, max_iter=100):
    iter_data = []
    ea = 100
    xr = a
    for i in range(max_iter):
        xr_old = xr
        xr = (a + b) / 2.0
        f_a = func(a)
        f_xr = func(xr)

        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100

        iter_data.append([i + 1, a, b, xr, ea, f_xr])

        if f_a * f_xr < 0:
            b = xr
        elif f_a * f_xr > 0:
            a = xr
        else:
            ea = 0
            break

        if ea < tol:
            break

    return xr, iter_data

def false_position_method(func, a, b, tol=0.001, max_iter=100):
    iter_data = []
    ea = 100
    xr = a
    for i in range(max_iter):
        xr_old = xr
        xr = b - (func(b) * (a - b)) / (func(a) - func(b))
        f_a = func(a)
        f_xr = func(xr)

        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100

        iter_data.append([i + 1, a, b, xr, ea, f_xr])

        if f_a * f_xr < 0:
            b = xr
        elif f_a * f_xr > 0:
            a = xr
        else:
            ea = 0
            break

        if ea < tol:
            break

    return xr, iter_data

def fixed_point_method(func, g_func, x0, tol=0.001, max_iter=100):
    iter_data = []
    ea = 100
    xr = x0
    for i in range(max_iter):
        xr_old = xr
        xr = g_func(xr_old)
        f_xr = func(xr)

        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100

        iter_data.append([i + 1, xr_old, xr, ea, f_xr])

        if ea < tol:
            break

    return xr, iter_data

def plot_function(func, a, b, root):
    x_vals = np.linspace(a - 1, b + 1, 400)
    y_vals = func(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(root, color='red', linestyle='--', label=f'Root at x={{root:.4f}}')
    plt.title('Function Plot')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print("Métodos Numéricos:")
    print("1. Método de la Bisección")
    print("2. Método de la Falsa Posición")
    print("3. Método del Punto Fijo")
    choice = input("Seleccione un método (1, 2, 3): ")

    user_input = input("Ingrese la función en términos de x (ej. x**3 - x - 2): ")
    a = float(input("Ingrese el valor de a (límite inferior): "))
    b = float(input("Ingrese el valor de b (límite superior): "))
    tol = float(input("Ingrese la tolerancia (ej. 0.001): "))

    x = symbols('x')
    func_expr = sympify(user_input)
    func = lambdify(x, func_expr, 'numpy')

    if choice == '1':
        root, iter_data = bisection_method(func, a, b, tol)
    elif choice == '2':
        root, iter_data = false_position_method(func, a, b, tol)
    elif choice == '3':
        g_input = input("Ingrese la función g(x) para el punto fijo: ")
        g_expr = sympify(g_input)
        g_func = lambdify(x, g_expr, 'numpy')
        x0 = float(input("Ingrese el valor inicial x0: "))
        root, iter_data = fixed_point_method(func, g_func, x0, tol)

    iter_df = pd.DataFrame(iter_data, columns=['Iteración', 'a', 'b', 'xr', 'ea (%)', 'f(xr)'])
    print("\nTabla de iteraciones:")
    print(iter_df)

    plot_function(func, a, b, root)

if __name__ == "__main__":
    main()
