import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis
import random

def random_sample(n_samples):
    """
    Takes random n_samples from a uniform distribution.
    Returns arrays x and y.
    """
    x = np.array([])
    y = np.array([])

    # Makes arrays of x and y values
    for i in range(n_samples):
        a = np.random.uniform(0,3.0)
        x = np.append(x,a-2)

        b = np.random.uniform(0,3.0)
        y = np.append(y,b-1.5)

    return x, y

def stratified_random_sample(n_samples):
    """
    Takes random n_samples from a linspace.
    Returns arrays x and shuffled y.
    """
    x = np.random.uniform(-2, 1, n_samples).tolist()
    y = np.linspace(-1.5, 1.5, n_samples)

    np.random.shuffle(y)

    x = np.array(x)
    y = np.array(y)

    return x, y