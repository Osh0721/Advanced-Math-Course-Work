import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np



#Question 5
#a
X=sym.symbols("X")
Y = np.linspace(-6, 6, 100)
EQ1 = 1/(1 + np.exp(-Y))
plt.plot(Y, EQ1)
plt.title('Logistic Function')
plt.show()

#b
Diff_EQ=EQ1*(1-EQ1)
plt.plot(Y,Diff_EQ)
plt.title('Derivative of Logistic Function')
plt.show()

#c
#a

X = np.linspace(-6, 6, 100)
Y = np.sin(np.sin(2*X))
plt.plot(X, Y)
plt.title('Plot of sin(np.sin(2*X))')
plt.show()


#b

X = np.linspace(-10, 10, 100)
Y = -X**3 - 2*X**2 + 3*X + 10
plt.plot(X, Y)
plt.title('Plot of -x^3 - 2x^2 + 3x + 10')
plt.show()



#c

X = np.linspace(-10, 10, 100)
Y = np.exp(-0.8*X)
plt.plot(X, Y)
plt.title('Plot of exp(-0.8*X)')
plt.show()

#d
X = np.linspace(-10, 10, 100)
Y =X**2*np.cos(np.cos(2*X)-2*np.sin(np.sin(X-(math.pi/3))))
plt.plot(X, Y)
plt.title('Plot of X**2*np.cos(np.cos(2*X)-2*np.sin(np.sin(X-(math.pi/3)))) ')
plt.show()


#e

x = np.linspace(-6, 6, 1000)
def A(x):
    if 0 <=x<np.pi:
        return x * np.exp(-0.4*x**2)

    if -np.pi<=x<0:
        return 2*np.cos(x+np.pi/6)

    if x>np.pi:
        P=x-(2*np.pi)
        R=A(P)
        return R

    if x<=-np.pi:
        P=x+(2*np.pi)
        R=A(P)
        return R

y=[A(l) for l in x]
plt.plot(x, y)
plt.show()

#d
# for a
X = np.linspace(-6, 6, 100)
Y = np.sin(np.sin(2*X))
Y_logistic = 1 / (1 + np.exp(-Y))
plt.plot(X, Y_logistic)
plt.title('Plot of logistic(sin(sin(2x)))')
plt.show()


# for b
X = np.linspace(-6, 6, 100)
Y = -X**3 - 2*X**2 + 3*X + 10
Y_logistic = 1 / (1 + np.exp(-Y))
plt.plot(X, Y_logistic)
plt.title('Plot of logistic -X**3 - 2*X**2 + 3*X + 10')
plt.show()

#for c
X = np.linspace(-6, 6, 100)
Y = np.exp(-0.8*X)
Y_logistic = 1 / (1 + np.exp(-Y))
plt.plot(X, Y_logistic)
plt.title('Plot of logistic np.exp(-0.8*X)')
plt.show()

#for d
X = np.linspace(-6, 6, 100)
Y =  X**2*np.cos(np.cos(2*X))-2*np.sin(np.sin(X-3.14/3))
Y_logistic = 1 / (1 + np.exp(-Y))
plt.plot(X, Y_logistic)
plt.title('Plot of logistc X**2*np.cos(np.cos(2*X)-2*np.sin(np.sin(X-(3.14/3))))')
plt.show()

