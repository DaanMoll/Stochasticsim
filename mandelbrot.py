import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

def random_sample(x, y, n_samples):
    # Makes a x and y array of random values
    for i in range(n_samples):
        a = np.random.uniform(-2,1)
        x = np.append(x,a)
        b = np.random.uniform(-1.5,1.5)
        y = np.append(y,b)
    return x, y

def lhs_sample(x, y, n_samples):
    space = Space([(-2.,1.), (-1.5, 1.5)])  
    lhs = Lhs(lhs_type="classic", criterion=None)
    coordinates = lhs.generate(space.dimensions, n_samples)

    for coordinate in coordinates:
        a = coordinate[0]
        x = np.append(x,a)
        b = coordinate[1]
        y = np.append(y,b)
    return x, y
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


def compute_area_mandelbrot(N_max, some_threshold, n_samples, kind):
    """
    Computes the Area of the mandel brot set with the monte carlo method.
    Takes as initial values N_max(integer), some_threshold(Number), nx(integer), ny(integer)
    returns the amount of hits and the ratio of hits/total shots
    """

    # Initializes values
    hits = 0
    x = np.array([])
    y = np.array([])
    n_samples

    if kind == "random":
        sample = random_sample(x, y, n_samples)
    elif kind == "lhs":
        sample = lhs_sample(x, y, n_samples)
    
    x = sample[0]
    y = sample[1]

    # Uses the random values for the mandelbrot computation
    mandelbrot_set = mandelbrot_computation(N_max, some_threshold, x, y)

    # Checks for all the values if they are in our out of the mandelbrot set
    for row in mandelbrot_set:
        for value in row:
            if value == True:
                hits+=1
    
    # calculates the ratio hits/total shots
    ratio = hits/n_samples**2

    return hits, ratio


if __name__=="__main__":
    
    print(compute_area_mandelbrot(100, 5, 100, "random"))
    print(compute_area_mandelbrot(100, 5, 100, "lhs"))




   