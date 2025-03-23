import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para graficar funciones
import pandas as pd  # Para manejar tablas de iteraciones
from sympy import symbols, lambdify, solve, sympify, diff  # Para manejo simbólico y solución de ecuaciones

# Método de Bisección para encontrar raíces de funciones
def bisection_method(func, a, b, tol=0.001, max_iter=100):
    iter_data = []  # Lista para almacenar datos de cada iteración
    ea = 100  # Error relativo inicial (alto para asegurar iteración)
    xr = a  # Valor inicial de xr

    for i in range(max_iter):
        xr_old = xr  # Guardamos el valor anterior de xr
        xr = (a + b) / 2.0  # Punto medio del intervalo
        f_a = func(a)
        f_xr = func(xr)

        # Cálculo del error relativo
        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100

        iter_data.append([i + 1, a, b, xr, ea, f_xr])  # Guardamos datos de la iteración

        # Determinamos el nuevo intervalo según el signo de f(a) * f(xr)
        if f_a * f_xr < 0:
            b = xr
        elif f_a * f_xr > 0:
            a = xr
        else:
            ea = 0  # Se encontró la raíz exacta
            break

        # Condición de parada
        if ea < tol:
            break

    return xr, iter_data  # Retorna la raíz estimada y la tabla de iteraciones

# Método de Falsa Posición
def false_position_method(func, a, b, tol=0.001, max_iter=100):
    iter_data = []
    ea = 100
    xr = a

    for i in range(max_iter):
        xr_old = xr
        xr = b - (func(b) * (a - b)) / (func(a) - func(b))  # Cálculo de xr con interpolación lineal
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

# Método de Newton-Raphson
def newton_raphson_method(func, dfunc, x0, tol=0.001, max_iter=100):
    iter_data = []
    ea = 100
    xr = x0

    for i in range(max_iter):
        xr_old = xr
        xr = xr_old - func(xr_old) / dfunc(xr_old)  # Fórmula de Newton-Raphson

        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100

        iter_data.append([i + 1, xr_old, xr, ea, func(xr)])

        if ea < tol:
            break

    return xr, iter_data

# Método del Punto Fijo
def fixed_point_method(func, g_funcs, x0, tol=0.001, max_iter=100):
    best_g = None
    best_iterations = max_iter
    best_root = None
    best_data = []

    for g_func in g_funcs:
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

        if len(iter_data) < best_iterations:
            best_iterations = len(iter_data)
            best_g = g_func
            best_root = xr
            best_data = iter_data

    return best_g, best_root, best_data

# Función para graficar f(x) y la raíz
def plot_function(func, a, b, root):
    x_vals = np.linspace(a - 1, b + 1, 400)
    y_vals = func(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(root, color='red', linestyle='--', label=f'Raíz en x={root:.4f}')
    plt.title('Gráfico de la Función')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función principal del programa
def main():
    print("Métodos Numéricos:")
    print("1. Método de la Bisección")
    print("2. Método de la Falsa Posición")
    print("3. Método de Newton-Raphson")
    print("4. Método del Punto Fijo")
    choice = input("Seleccione un método (1, 2, 3, 4): ")

    user_input = input("Ingrese la función en términos de x (ej. x**3 - x - 2): ")
    x = symbols('x')
    func_expr = sympify(user_input)
    func = lambdify(x, func_expr, 'numpy')
    g_expressions = solve(x - func_expr, x)
    g_funcs = [lambdify(x, g_expr, 'numpy') for g_expr in g_expressions]

    if choice in ['1', '2']:
        a = float(input("Ingrese el valor de a: "))
        b = float(input("Ingrese el valor de b: "))
        tol = float(input("Ingrese la tolerancia: "))
        root, iter_data = (bisection_method if choice == '1' else false_position_method)(func, a, b, tol)
    elif choice == '3':
        x0 = float(input("Ingrese el valor inicial x0: "))
        tol = float(input("Ingrese la tolerancia: "))
        dfunc = lambdify(x, diff(func_expr, x), 'numpy')
        root, iter_data = newton_raphson_method(func, dfunc, x0, tol)
    elif choice == '4':
        x0 = float(input("Ingrese el valor inicial x0: "))
        tol = float(input("Ingrese la tolerancia: "))
        best_g, root, iter_data = fixed_point_method(func, g_funcs, x0, tol)
    else:
        print("Opción no válida")
        return

    iter_df = pd.DataFrame(iter_data, columns=['Iteración', 'a', 'b', 'xr', 'ea (%)', 'f(xr)'] if choice in ['1', '2'] else ['Iteración', 'x_old', 'x_new', 'ea (%)', 'f(xr)'])
    print("\nTabla de iteraciones:")
    print(iter_df)
    plot_function(func, a if choice in ['1', '2'] else x0 - 1, b if choice in ['1', '2'] else x0 + 1, root)

if __name__ == "__main__":
    main()
