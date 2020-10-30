import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis

def compute_mandelbrot(N_max, some_threshold, nx, ny):
    # A grid of c-values
    x = np.linspace(-2, 1, nx)
    y = np.linspace(-1.5, 1.5, ny)


    print(x[:,newaxis])
    c = x[:,newaxis] + 1j*y[newaxis,:]

    print(c)
    # Mandelbrot iteration
    z = c

    # The code below overflows in many regions of the x-y grid, suppress
    # warnings temporarily
    for j in range(N_max):
        z = z**2 + c
    print(abs(z))
    mandelbrot_set = (abs(z) < some_threshold)

    return mandelbrot_set

mandelbrot_set = compute_mandelbrot(50, 5., 1000, 1000)

plt.imshow(mandelbrot_set.T, extent=[-2, 1, -1.5, 1.5])
plt.gray()
plt.show()