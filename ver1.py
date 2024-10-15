# Ejercicio 2 Parcial 4 Métodos Númericos

#librerias
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import solve
from math import e
#Parámetros del problema

#x



#solución exacta
def solucion_exacta(x):
    return 1 / 6 * (x**3 * e**x) - 5 / 3 * (x * e**x) + 2 * e**x - x - 2

def solucion_bvp(x):
    def bc(Ya, Yb):
        return np.array([Ya[0], Yb[0]])
    xg= np.linspace(0,100)
    inicialg=np.zeros((2,xg.size))
    def funcion(x,y):
        
        return np
    solbb= solve_bvp(funcion, bc, xg, inicialg)
    return 

# Método de diferencias finitas
def diferencias_finitas(n_puntos):
    # Calcular el paso de discretización (dx)
    dx = L / (n_puntos + 1)
    
    # Crear un arreglo de puntos x con n_puntos + 2 elementos (incluyendo los extremos)
    x = np.linspace(0, L, n_puntos + 2)
    
    # Crear una matriz A y un vector b para el sistema de ecuaciones lineales
    A = np.zeros((n_puntos, n_puntos))
    b = np.zeros(n_puntos)
    
    # Rellenar la matriz A y el vector b
    for i in range(n_puntos):
        xi = (i + 1) * dx  # Calcular el valor de x en el punto i
        A[i, i] = -2 / dx**2  # Elemento diagonal de la matriz A
        if i > 0:
            A[i, i-1] = 1 / dx**2  # Elemento subdiagonal de la matriz A
        if i < n_puntos - 1:
            A[i, i+1] = 1 / dx**2  # Elemento superdiagonal de la matriz A
        b[i] = (w * L * xi / (2 * E * I)) - (w * xi**2 / (2 * E * I))  # Elemento del vector b
    
    # Resolver el sistema de ecuaciones lineales
    y_finitas = solve(A, b)
    
    # Agregar los valores de las condiciones de frontera (y(0) = y(L) = 0)
    y_finitas = np.concatenate(([0], y_finitas, [0]))
    
    return x, y_finitas  # Devolver la solución encontrada

# Graficar los resultados
x_exacta = np.linspace(0, L, 100)
y_exacta = solucion_exacta(x_exacta)
x_finitas, y_finitas = diferencias_finitas(50) #aquí el 50, es el valor de los puntos o nodos

