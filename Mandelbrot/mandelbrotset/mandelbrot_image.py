import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

from .mandelbrotcompute import mandelbrot_computation

def compute_image_mandelbrot(N_max, some_threshold, nx=601, ny=201):
    """
    Computes an image of the mandel brot set.
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
    plt.title("Computed image of the Mandelbrot set")
    plt.xlabel("x-coordinates")
    plt.ylabel("y-coordinates")
    plt.show()