import numpy as np
import scipy.fftpack as sfft
import matplotlib.pyplot as plt
#Question2
X = np.arange(-np.pi,np.pi, 0.0001)
X1 = np.arange(-np.pi,np.pi, 0.1)

Y = np.sin(2*X) + 0.5*np.sin(70*X)
Y1 = np.sin(2*X1) + 0.5*np.sin(70*X1)


Yf = sfft.fft(Y)
Yf1 = sfft.fft(Y1)

plt.plot(X,Y)
plt.plot(X1,Y1)
plt.legend(["10000Hz","10Hz"])
plt.show()

#alising happens in the low frequncy (10hz)