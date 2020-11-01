import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis

def mandelbrot_computation(N_max, some_threshold, x, y):
    """
    Computes the mandelbrot set. 
    Takes as initial values: N_max(integer), some_threshold(number), x(array), y(array)
    Returns a array with boolean values, True when value is part of the mandelbrot set and False if not.
    """
    c = x[:,newaxis] + 1j*y[newaxis,:]

    # Mandelbrot iteration
    z = c

    # Does the mandelbrot iteration N_max times
    for j in range(N_max):
        z = z**2 + c
    
    # Checks for all the values if value is part of the mandelbrot set
    mandelbrot_set = (abs(z) < some_threshold)

    return mandelbrot_set

def compute_image_mandelbrot(N_max, some_threshold, nx, ny):
    """
    Computes a image of the mandel brot set.
    Takes as initial values N_max(integer), some_threshold(Number), nx(integer), ny(integer)
    """
    # A grid of c-values
    x = np.linspace(-2, 1, nx)
    y = np.linspace(-1.5, 1.5, ny)
    
    # Makes complex number wit x and y values
    c = x[:,newaxis] + 1j*y[newaxis,:]

    # Computes the mandelbrot with x and y values
    mandelbrot_set = mandelbrot_computation(N_max, some_threshold, x, y)

    # Makes image of mandelbrot set
    plt.imshow(mandelbrot_set.T, extent=[-2, 1, -1.5, 1.5])
    plt.gray()
    plt.show()


def compute_area_mandelbrot(N_max, some_threshold, nx, ny):
    """
    Computes the Area of the mandel brot set with the monte carlo method.
    Takes as initial values N_max(integer), some_threshold(Number), nx(integer), ny(integer)
    returns the amount of hits and the ratio of hits/total shots
    """
    # Initializes values
    hits = 0
    x = np.array([])
    y = np.array([])

    # Makes a x and y array of random values
    for i in range(nx):
        a = np.random.uniform(-2,1)
        x = np.append(x,a)
    for i in range(ny):
        b = np.random.uniform(-1.5,1.5)
        y = np.append(y,b)

    # Uses the random values for the mandelbrot computation
    mandelbrot_set = mandelbrot_computation(N_max, some_threshold, x, y)

    # Checks for all the values if they are in our out of the mandelbrot set
    for row in mandelbrot_set:
        for value in row:
            if value == True:
                hits+=1
    
    # calculates the ratio hits/total shots
    ratio = hits/(nx*ny)

    return hits, ratio

if __name__=="__main__":
    
    monte_carlo = compute_area_mandelbrot(50, 5., 100, 10)
    hits = monte_carlo[0]
    ratio = monte_carlo[1]

    print(hits, ratio)
    compute_image_mandelbrot(50, 5., 100, 10)
