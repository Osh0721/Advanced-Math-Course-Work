import scipy.fftpack as sfft
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import scipy.signal as Signal


# a)

image = mpimg.imread(r"Fruit (1).jpg")
img_fft = sfft.fft2(image)
plt.imshow(np.abs(img_fft))
imgf = sfft.fftshift(img_fft)
plt.imshow(np.abs(imgf))


imgf1 = np.zeros((360, 360), dtype=complex)
c = 180
r = 90
for m in range(0, 360):
    for n in range(0, 360):
        if np.sqrt(((m - c) ** 2 + (n - c) ** 2)) > r:
            imgf1[m, n] = imgf[m, n]

Last_img = sfft.ifft2(imgf1)
plt.imshow(np.abs(Last_img))
plt.show()

# b)
sigma1=5
sigma2=5
kernel = np.outer(np.abs(Signal.gaussian(360, sigma1)), np.abs(Signal.gaussian(360, sigma2)))
kernel_f = sfft.fft2(sfft.ifftshift(kernel))
image_f = sfft.fft2(image)
blurred_image_f = image_f * kernel_f
blurred_image = sfft.ifft2(blurred_image_f)
new_image=np.abs(blurred_image)
plt.imshow(new_image)
plt.show()

#c)

imgc = sfft.dct((sfft.dct(image, norm='ortho')).T, norm='ortho')
plt.imshow(imgc)
image1 = sfft.idct((sfft.idct(image, norm='ortho')).T, norm='ortho')
imgc2 = imgc[0:240, 0:240]
image1 = sfft.idct((sfft.idct(imgc2, norm='ortho')).T, norm='ortho')
plt.imshow(image1)
plt.show()

#d)
image1 = np.zeros((480, 480))
image1[:120, :120] = imgc[:120, :120]
img1 = sfft.idct((sfft.idct(image1, norm='ortho')).T, norm='ortho')
plt.imshow(img1)
plt.show()