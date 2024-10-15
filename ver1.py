# Ejercicio 2 Parcial 4 Métodos Númericos

'''
## Tema B

Se plantea la ecuación:

$$
y'' = 2y' - y + xe^x - x
$$

Con condiciones de 

- y(0) = 0
- y(2) = - 4
- 0 $\le$ x $\le$ 2


Y en el ejercicio se da la solución exacta:

$$
y(x) = frac{1}{6}(x^3e^x) - frac{5}{3}(xe^x) + 2e^x - x -2
$$
'''



#librerias
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def ecuacion(x, y):
    return np.array([y[1], 2*y[1] - y[0] + x*np.exp(x) - x])

def condicion_frontera(ya, yb):
    return np.array([ya[0], yb[0] - (-4)])

#solución exacta
def solucion_exacta(x):
    return (1/6)*x**3*np.exp(x) - (5/3)*x*np.exp(x) + 2*np.exp(x) - x - 2

def diferencias_finitas(n_puntos):
    L = 2.0
    h = L / (n_puntos + 1)  # Tamaño del paso
    x = np.linspace(0, L, n_puntos + 2)  # Puntos x
    y = np.zeros(n_puntos + 2)  # Inicializar el vector de soluciones
    y[0] = 0  # Condición de frontera y(0) = 0
    y[-1] = -4  # Condición de frontera y(2) = -4
    
    A = np.zeros((n_puntos, n_puntos))  # Matriz del sistema
    b = np.zeros(n_puntos)  # Vector del lado derecho

    for i in range(1, n_puntos + 1):
        xi = x[i]  # Punto actual
        # Construcción de la matriz A
        A[i-1, i-1] = 2 + h  # Coeficiente de y[i]
        if i > 1:
            A[i-1, i-2] = -1  # Coeficiente de y[i-1]
        if i < n_puntos:
            A[i-1, i] = -1  # Coeficiente de y[i+1]
        
        # Construcción del vector b
        b[i-1] = h**2 * (xi * np.exp(xi) - xi)  # Término independiente

    # Resolver el sistema de ecuaciones
    y[1:-1] = np.linalg.solve(A, b)
    
    return x, y

# Graficar los resultados
x_exacta = np.linspace(0, 2, 100)
y_exacta = solucion_exacta(x_exacta)

x_finitas, y_finitas = diferencias_finitas(50)

result = solve_bvp(ecuacion, condicion_frontera, np.linspace(0, 2, 100), np.zeros((2, 100)))
x_bvp = result.x
y_bvp = result.y

plt.scatter(x_exacta, y_exacta, label='Solución exacta',  color='red')
plt.plot(x_bvp, y_bvp[0], label='Solución BVP')
plt.plot(x_finitas, y_finitas, label='Solución diferencias finitas')
plt.legend()
plt.show()

# Calcular error local y global
n_puntos = 50
x, y_finitas = diferencias_finitas(n_puntos)
x_exacta = np.linspace(0, 2, 100)
y_exacta = solucion_exacta(x_exacta)

error_local = np.abs(y_finitas - solucion_exacta(x))
error_global = np.linalg.norm(error_local) / np.sqrt(n_puntos)

# Graficar error local
plt.plot(x, error_local)
plt.xlabel('x')
plt.ylabel('Error local')
plt.title('Error local entre la solución exacta y la solución numérica')
plt.show()

# Graficar error global
plt.plot([n_puntos], [error_global], 'o')
plt.xlabel('Número de puntos')
plt.ylabel('Error global')
plt.title('Error global entre la solución exacta y la solución numérica')
plt.show()