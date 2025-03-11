import numpy as np  # Para operaciones numéricas y generación de datos
import matplotlib.pyplot as plt  # Para graficar funciones
import pandas as pd  # Para manejar tablas de iteraciones
from sympy import symbols, lambdify, sympify  # Para manejar expresiones matemáticas simbólicas


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


# Método del Punto Fijo

def fixed_point_method(func, g_func, x0, tol=0.001, max_iter=100):
    iter_data = []
    ea = 100
    xr = x0  # Punto inicial

    for i in range(max_iter):
        xr_old = xr
        xr = g_func(xr_old)  # Aplicación de la función g(x)
        f_xr = func(xr)

        if i > 0:
            ea = abs((xr - xr_old) / xr) * 100

        iter_data.append([i + 1, xr_old, xr, ea, f_xr])

        if ea < tol:
            break

    return xr, iter_data


# Función para graficar la función ingresada

def plot_function(func, a, b, root):
    x_vals = np.linspace(a - 1, b + 1, 400)  # Generación de valores x en un rango extendido
    y_vals = func(x_vals)  # Evaluación de la función en x_vals

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='f(x)')  # Graficamos f(x)
    plt.axhline(0, color='black', linewidth=1)  # Línea horizontal en y=0
    plt.axvline(root, color='red', linestyle='--', label=f'Raíz en x={root:.4f}')  # Línea vertical en la raíz
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
    choice = input("Seleccione un método (1, 2, 3): ")

    # Solicitar datos de entrada al usuario
    user_input = input("Ingrese la función en términos de x (ej. x**3 - x - 2): ")
    a = float(input("Ingrese el valor de a (límite inferior): "))
    b = float(input("Ingrese el valor de b (límite superior): "))
    tol = float(input("Ingrese la tolerancia (ej. 0.001): "))

    x = symbols('x')  # Definimos la variable simbólica
    func_expr = sympify(user_input)  # Convertimos la cadena en una expresión simbólica
    func = lambdify(x, func_expr, 'numpy')  # Convertimos la expresión en función evaluable

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
    else:
        print("Opción no válida")
        return

    # Mostrar la tabla de iteraciones en un DataFrame
    iter_df = pd.DataFrame(iter_data, columns=['Iteración', 'a', 'b', 'xr', 'ea (%)', 'f(xr)'])
    print("\nTabla de iteraciones:")
    print(iter_df)

    # Graficar la función y la raíz encontrada
    plot_function(func, a, b, root)


# Ejecutar el programa si se ejecuta este script directamente
if __name__ == "__main__":
    main()