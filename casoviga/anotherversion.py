import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve

# Parámetros del problema
E_ksi = 30  # Módulo de elasticidad en Ksi
E = E_ksi * 1000  # Conversión de Ksi a psi
I = 800  # Momento de inercia en in^4
w_kip_ft = 1  # Carga distribuida en kip/ft
w = w_kip_ft * 1000 / 12  # Conversión de kip/ft a lb/in
L_ft = 10  # Longitud de la viga en pies
L = L_ft * 12  # Conversión de pies a pulgadas

print(f"Módulo de elasticidad E: {E} psi")
print(f"Momento de inercia I: {I} in^4")
print(f"Carga distribuida w: {w} lb/in")
print(f"Longitud de la viga L: {L} in")

# Solución exacta
def solucion_exacta(x):
    return (w * L * x**3) / (12 * E * I) - (w * x**4) / (24 * E * I) - (w * L**3 * x) / (24 * E * I)

#método disparo lineal
def disparo_lineal():
    # Definir el sistema de  ecuaciones diferenciales
    def sistema_ecuaciones(x, y):
        # y[0] es la función desconocida (desplazamiento) y y[1] es su derivada (velocidad)
        # La ecuación diferencial es: y'' = (w * L * x / (2 * E * I)) - (w * x**2 / (2 * E * I))
        return [y[1], (w * L * x / (2 * E * I)) - (w * x**2 / (2 * E * I))]
    
    # Encontrar la condición inicial óptima mediante bisección
    a, b = -1, 1  # Intervalo inicial para buscar la condición inicial óptima
    tol = 1e-6  # Tolerancia para detener la búsqueda
    while b - a > tol:
        y_prime_0 = (a + b) / 2  # Punto medio del intervalo actual
        sol = solve_ivp(sistema_ecuaciones, [0, L], [0, y_prime_0], t_eval=np.linspace(0, L, 100))
        # Verificar si la solución encontrada satisface la condición de frontera en x=L
        if sol.y[0][-1] > 0:
            b = y_prime_0  # Si la condición no se cumple, reducir el intervalo hacia abajo
        else:
            a = y_prime_0  # Si la condición se cumple, reducir el intervalo hacia arriba
    
    # Resolver con la condición inicial óptima encontrada
    sol = solve_ivp(sistema_ecuaciones, [0, L], [0, y_prime_0], t_eval=np.linspace(0, L, 100))
    return sol.t, sol.y[0]  # Devolver la solución encontrada

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

x_disparo, y_disparo = disparo_lineal()
x_finitas, y_finitas = diferencias_finitas(50) #aquí el 50, es el valor de los puntos o nodos

# Graficar todas las soluciones
plt.figure(figsize=(10, 6))
plt.plot(x_exacta, y_exacta, label="Solución Exacta", color="black", linewidth=2)
plt.plot(x_disparo, y_disparo, label="Método del Disparo Lineal", linestyle="--", color="blue")
plt.plot(x_finitas, y_finitas, label="Método de Diferencias Finitas", linestyle=":", color="red")
plt.xlabel("x (in)")
plt.ylabel("y (in)")
plt.title("Comparación de Métodos para la Deflexión de la Viga")
plt.legend()
plt.grid(True)
plt.show()
