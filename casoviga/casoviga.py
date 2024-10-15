# Librerias
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Parámetros del problema
E = 30 * 1000 * 1000  # Pasar de Ksi a psi (módulo de elasticidad)
I = 800  # en in^4 (momento de inercia de la viga)
w = 1 * 1000 / 12  # pasar de kip/ft a lb/in (carga distribuida)
L = 10 * 12  # pasar de pies a pulgadas (longitud de la viga)
n = 50  # Número de nodos en la malla

# Resolver por Método de disparo lineal
# 1 Definir la ecuación diferencial
def ecuacion(x, y):
    return [y[1], (w * L * x - w * x**2) / (2 * E * I)]  # y[1] es la derivada de la deflexión

# Condiciones de frontera
def c_frontera(y1, y2):
    return [y1[0], y2[0]]  # y(0) = 0 y y(L) = 0

# Arreglo de puntos
t_puntos = np.linspace(0, L, 100)

# Inicializar las condiciones de frontera
y0 = np.zeros((2, len(t_puntos)))
y0[0, 0] = 0  # Condición inicial y(0) = 0
y0[1, 0] = 1  # Condición inicial arbitraria para y'(0)

def disparo_lineal():  # Resolver utilizando solve_bvp
    sln = solve_bvp(ecuacion, c_frontera, t_puntos, y0)
    return sln.y[0]

resultados_disparo = disparo_lineal()

def dif_finitas(n):
    x_finitas = np.linspace(0, L, n + 1)
    dx = x_finitas[1] - x_finitas[0]

    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    for i in range(1, n):
        A[i, i-1] = 1 / dx**2 #inferior
        A[i, i] = -2 / dx**2 #principal
        A[i, i+1] = 1 / dx**2 #superior
        b[i] = (w * x_finitas[i] * (L - x_finitas[i])) / (2 * E * I)

    A[0, 0] = 1
    A[n, n] = 1
    b[0] = 0
    b[n] = 0

    sln_finitas = np.linalg.solve(A, b)

    return x_finitas, sln_finitas

# Solución exacta dada en el problema para la deflexión de la viga
def sln_exacta(x):
    return (w * L * x**3) / (12 * E * I) - (w * x**4) / (24 * E * I) - (w * L**3 * x) / (24 * E * I)

# Gráfica
x_disparo = t_puntos
y_disparo = resultados_disparo

x_finitas, y_finitas = dif_finitas(n)

x_exacta = np.linspace(0, L, 100)
y_exacta = sln_exacta(x_exacta)

plt.plot(x_exacta, y_exacta, label="Solución Exacta", color="r")
plt.plot(x_finitas, y_finitas, label="sln Diferencias Finitas", color="green")
plt.plot(x_disparo, y_disparo, label="sln Disparo Lineal", linestyle="--", color="b")
plt.title("Deflexión de la Viga")
plt.legend()
plt.grid(True)
plt.show()
