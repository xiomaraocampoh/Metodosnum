import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para graficar funciones
import pandas as pd  # Para manejar tablas de iteraciones
from sympy import symbols, lambdify, diff, solve, sympify  # Para manejo simbólico y cálculo de derivadas

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


# Método del Punto Fijo corregido
def fixed_point_method(func, g_func, x0, tol=0.001, max_iter=100):
    iter_data = []  # Datos de iteración
    ea = 100  # Error relativo inicial alto
    xr = x0  # Punto inicial

    for i in range(max_iter):
        xr_old = xr  # Guardar valor anterior de xr
        xr = g_func(xr_old)  # Aplicar g(x)
        f_xr = func(xr)  # Evaluar f(x) en xr

        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100  # Calcular error relativo

        iter_data.append([i + 1, xr_old, xr, ea, f_xr])  # Guardar datos de iteración

        if ea < tol:  # Verificar criterio de convergencia
            break

    return xr, iter_data  # Retorna la raíz y los datos de iteración


# Método de Newton-Raphson
def newton_raphson_method(func, func_prime, x0, tol=0.001, max_iter=100):
    iter_data = []  # Datos de iteración
    ea = 100  # Error relativo inicial alto
    xr = x0  # Punto inicial

    for i in range(max_iter):
        xr_old = xr  # Guardar valor anterior de xr
        xr = xr_old - func(xr_old) / func_prime(xr_old)  # Fórmula de Newton-Raphson
        f_xr = func(xr)  # Evaluar f(x) en xr

        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100  # Calcular error relativo

        iter_data.append([i + 1, xr_old, xr, ea, f_xr])  # Guardar datos de iteración

        if ea < tol:  # Verificar criterio de convergencia
            break

    return xr, iter_data  # Retorna la raíz y los datos de iteración


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
    print("3. Método del Punto Fijo")
    print("4. Método de Newton-Raphson")
    choice = input("Seleccione un método (1, 2, 3, 4): ")

    user_input = input("Ingrese la función en términos de x (ej. x**3 - x - 2): ")
    x = symbols('x')
    func_expr = sympify(user_input)
    func = lambdify(x, func_expr, 'numpy')
    func_prime = lambdify(x, diff(func_expr, x), 'numpy')

    if choice in ['1', '2']:
        a = float(input("Ingrese el valor de a (límite inferior): "))
        b = float(input("Ingrese el valor de b (límite superior): "))
        tol = float(input("Ingrese la tolerancia (ej. 0.001): "))
        root, iter_data = bisection_method(func, a, b, tol) if choice == '1' else false_position_method(func, a, b, tol)
    elif choice == '3':
        x0 = float(input("Ingrese el valor inicial x0: "))
        tol = float(input("Ingrese la tolerancia (ej. 0.001): "))
        g_expr = solve(x - func_expr, x)[0]
        g_func = lambdify(x, g_expr, 'numpy')
        root, iter_data = fixed_point_method(func, g_func, x0, tol)
    elif choice == '4':
        x0 = float(input("Ingrese el valor inicial x0: "))
        tol = float(input("Ingrese la tolerancia (ej. 0.001): "))
        root, iter_data = newton_raphson_method(func, func_prime, x0, tol)
    else:
        print("Opción no válida")
        return

    print(pd.DataFrame(iter_data))
    plot_function(func, x0 - 1, x0 + 1, root)

if __name__ == "__main__":
    main()
