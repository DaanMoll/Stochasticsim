import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

from .sampling.lhs import lhs_sample
from .sampling.ortho import ortho_sample 
from .sampling.random_sampling import random_sample
from .sampling.orthogonal import scale_points

from .mandelbrotcompute import mandelbrot_computation

def compute_area_mandelbrot(N_max, some_threshold, n_samples, kind):
    """
    Computes the Area of the mandel brot set with the monte carlo method.
    Takes as initial values N_max(integer), some_threshold(Number), n_samples(integer), kind(string = random, lhs, ortho)
    returns the amount of hits and the ratio of hits/total shots
    """

    # Initializes values
    hits = 0
    x = np.array([])
    y = np.array([])

    if kind == "Random":
        sample = random_sample(x, y, n_samples)
    elif kind == "LHS":
        sample = lhs_sample(x, y, n_samples)
    elif kind == "Orthogonal":
        sample = scale_points(n_samples)
    
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

    # total plot area is 9 so ratio * 9
    area = ratio * 9

    return hits, ratio, area
