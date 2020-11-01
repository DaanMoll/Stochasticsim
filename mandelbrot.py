import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis

def mandelbrot_computation(N_max, some_threshold, x, y):
    c = x[:,newaxis] + 1j*y[newaxis,:]

    # Mandelbrot iteration
    z = c

    # The code below overflows in many regions of the x-y grid, suppress
    # warnings temporarily
    for j in range(N_max):
        z = z**2 + c
    
    mandelbrot_set = (abs(z) < some_threshold)
    return mandelbrot_set

def compute_image_mandelbrot(N_max, some_threshold, nx, ny):
    # A grid of c-values
    x = np.linspace(-2, 1, nx)
    y = np.linspace(-1.5, 1.5, ny)
    
    c = x[:,newaxis] + 1j*y[newaxis,:]

    mandelbrot_set = mandelbrot_computation(N_max, some_threshold, x, y)
    plt.imshow(mandelbrot_set.T, extent=[-2, 1, -1.5, 1.5])
    plt.gray()
    plt.show()


def compute_area_mandelbrot(N_max, some_threshold, nx, ny):
    # A grid of c-values
    hits = 0

    x = np.array([])
    y = np.array([])

    for i in range(nx):
        a = np.random.uniform(-2,1)
        x = np.append(x,a)
    for i in range(ny):
        b = np.random.uniform(-1.5,1.5)
        y = np.append(y,b)

    mandelbrot_set = mandelbrot_computation(N_max, some_threshold, x, y)
    for row in mandelbrot_set:
        for value in row:
            if value == True:
                hits+=1
    
    ratio = hits/(nx*ny)

    return hits, ratio

if __name__=="__main__":
    
    monte_carlo = compute_area_mandelbrot(50, 5., 100, 10)
    hits = monte_carlo[0]
    ratio = monte_carlo[1]

    print(hits, ratio)
    compute_image_mandelbrot(50, 5., 100, 10)
