import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Parámetros del problema
E = 30 * 1000  # Conversión de Ksi a psi (módulo de elasticidad)
I = 800  # Momento de inercia en in^4 (momento de inercia de la viga)
w = 1 * 1000 / 12  # Conversión de kip/ft a lb/in (carga distribuida)
L = 10 * 12  # Conversión de pies a pulgadas (longitud de la viga)

# Solución exacta para la deflexión de la viga
def solucion_exacta(x):
    # Fórmula para la deflexión de la viga bajo carga distribuida
    return (w * L * x**3) / (12 * E * I) - (w * x**4) / (24 * E * I) - (w * L**3 * x) / (24 * E * I)

# Método de disparo lineal
def disparo_lineal():
    # Definir la ecuación diferencial que describe la deflexión de la viga
    def ecuacion(t, y):
        # y[0] es la deflexión y y[1] es la derivada de la deflexión
        return [y[1], (w * L * t - w * t**2) / (2 * E * I)]

    # Definir las condiciones de frontera
    def condiciones_frontera(ya, yb):
        return [ya[0], yb[0]]  # y(0) = 0 y y(L) = 0

    # Crear un arreglo de puntos en el dominio [0, L]
    t_eval = np.linspace(0, L, 100)

    # Inicializar la condición inicial como una matriz 2D
    y0 = np.zeros((2, len(t_eval)))
    y0[0, 0] = 0  # Condición inicial y(0) = 0
    y0[1, 0] = 1  # Condición inicial arbitraria para y'(0)

    # Resolver el problema de valor de frontera utilizando solve_bvp
    sol = solve_bvp(ecuacion, condiciones_frontera, t_eval, y0)

    # Devolver los valores de x y y
    return sol.x, sol.y[0]

# Método de diferencias finitas
def diferencias_finitas(n_puntos):
    # Calcular el paso de discretización
    dx = L / (n_puntos + 1)

    # Crear un arreglo de puntos en el dominio [0, L]
    x = np.linspace(0, L, n_puntos + 2)

    # Crear la matriz A y el vector b para el sistema de ecuaciones lineales
    A = np.diag([-2] * n_puntos) + np.diag([1] * (n_puntos - 1), -1) + np.diag([1] * (n_puntos - 1), 1)
    b = [(w * L * xi - w * xi**2) / (2 * E * I) for xi in x[1:-1]]

    # Resolver el sistema de ecuaciones
    y_finitas = np.linalg.solve(A / dx**2, b)

    # Agregar los valores de las condiciones de frontera y(0) = y(L) = 0
    return x, np.concatenate(([0], y_finitas, [0]))

# Graficar los resultados
x_exacta = np.linspace(0, L, 100)
y_exacta = solucion_exacta(x_exacta)

x_disparo, y_disparo = disparo_lineal()
x_finitas, y_finitas = diferencias_finitas(50)

plt.figure(figsize=(10, 6))
plt.plot(x_exacta, y_exacta, label="Solución Exacta", color="black", linewidth=2)
plt.plot(x_disparo, y_disparo, label="Disparo Lineal", linestyle="--", color="blue")
plt.plot(x_finitas, y_finitas, label="Diferencias Finitas", linestyle=":", color="red")
plt.xlabel("x (in)")
plt.ylabel("y (in)")
plt.title("Comparación de Métodos para la Deflexión de la Viga")
plt.legend()
plt.grid(True)
plt.show()