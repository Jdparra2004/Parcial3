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
    dx = L / (n_puntos + 1)
    x = np.linspace(0, L, n_puntos + 2)
    y = np.zeros(n_puntos + 2)
    y[0] = 0
    y[-1] = -4
    
    A = np.zeros((n_puntos, n_puntos))
    b = np.zeros(n_puntos)
    
    for i in range(n_puntos):
        xi = (i + 1) * dx
        A[i, i] = -2 / dx**2
        if i > 0:
            A[i, i-1] = 1 / dx**2
        if i < n_puntos - 1:
            A[i, i+1] = 1 / dx**2
        b[i] = xi*np.exp(xi) - xi
    
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