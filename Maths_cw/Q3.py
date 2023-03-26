import math
import sympy
import matplotlib.pyplot as plt
import numpy as np

#Question3
#a
X= np.linspace(-15, 21, 100)
Y=X*np.cos(X/2)
plt.plot(X, Y)
plt.title('Plot of f(x)')
plt.show()

#b

x = np.pi / 2
n = 10
x= x % (2 * np.pi)
Total = 0
for i in range(0, n + 1):
    Total += ((-1) * i) * (x * (2 * i)) / math.factorial(2 * i)
print("answer is ",Total)
#c

def taylor_series_cosine(x, x0, count):
    series = sympy.series(sympy.cos(x), x0, count)
    range_of_function = np.linspace(-10, 10, 100)
    y = [series.evalf(subs={x: x_val}) for x_val in range_of_function]
    plt.title("Taylor Series")
    plt.plot(range_of_function, y)
    plt.show()

taylor_series_cosine(sympy.Symbol('x'), np.pi / 2, 60)


#d

def taylor_cos(x, a, n):
    series = 0
    for i in range(n+1):
        series += (-1)**i * (x-a)**(2*i) / np.math.factorial(2*i)
    return series


approximation = np.pi/3 * taylor_cos(np.pi/6, np.pi/2, 60)

print("Approximation:", approximation)
print("Actual value: ", np.pi/3 * np.cos(np.pi/6))


