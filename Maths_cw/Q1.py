import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np

#Quetion1

#a
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if x < 0 and x >= -np.pi:
        return x**2+1
    elif x >= 0 and x <= np.pi:
        return x*np.exp(-x)
    elif x<-np.pi:
        A=x+(2*np.pi)
        R=f(A)
        return R
    elif x>np.pi:
        A =x-(2*np.pi)
        R =f(A)
        return R


x_value = np.linspace(-4*np.pi,4*np.pi,1000)
Y= [f(b) for b in x_value]
plt.plot(x_value, Y)
plt.title('Periodic function f(x)')
plt.show()

#b

x = sym.symbols('x')
n = sym.symbols('n', integer=True, positive=True)

ms = np.empty(150, dtype=object)
xrange = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
y = np.zeros([151, 1000])

eq = (x ** 2) + 1
eq2 = x * sym.exp(-1 * x)

a0 = (1 / (2 * sym.pi))*(eq.integrate((x,-1*sym.pi,0))+eq2.integrate((x,0,sym.pi)))
print("a0 value is:",a0)
an = (1 / sym.pi)*(sym.integrate((eq*sym.cos(n*x)),(x,-1*sym.pi,0))+sym.integrate((eq2*sym.cos(n*x)),(x, 0, sym.pi)))
print("an value is",an)
bn = (1 / sym.pi) * (sym.integrate((eq * sym.sin(n * x)), (x,-1*sym.pi,0)) + sym.integrate((eq2*sym.sin(n*x)),(x, 0, sym.pi)))
print("bn value is",bn)

ms[0]=a0
f = sym.lambdify(x, ms[0], 'numpy')
y[0,:] = f(xrange)

for m in range(1, 150):
    ms[m] = ms[m-1] + an.subs(n, m) * sym.cos(m * x) + bn.subs(n, m) * sym.sin(m * x)
    f = sym.lambdify(x, ms[m], 'numpy')
    y[m, :] = f(xrange)

print("Fourier series is",ms[1])

#c


plt.plot(xrange, y[1, :], label="Up to 1st harmonic")
plt.plot(xrange, y[4, :], label="Up to 5th harmonic")
plt.plot(xrange, y[149, :], label="Up to 150th harmonic")
plt.plot(xrange, y[150, :], label="Original function")
plt.legend()
plt.show()

#d
actual_value = [y[1, :]]
predicted_value = [Y]

def calculate_rmse(harmonic_values, predicted_values):
    mse = np.square(np.subtract(harmonic_values, predicted_values)).mean()
    rmse = math.sqrt(mse)
    return rmse

first_harmonic_rmse = calculate_rmse([y[1, :]], [Y])
print("Root Mean Square Error for 1st harmonic:", first_harmonic_rmse)

fifth_harmonic_rmse = calculate_rmse([y[4, :]], [Y])
print("Root Mean Square Error for 5th harmonic:", fifth_harmonic_rmse)

one_fifty_harmonic_rmse = calculate_rmse([y[149, :]], [Y])
print("Root Mean Square Error for 150th harmonic:", one_fifty_harmonic_rmse)

#RMSE decreases from 1th harmonic to 150th harmonic