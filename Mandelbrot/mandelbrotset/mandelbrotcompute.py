import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
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