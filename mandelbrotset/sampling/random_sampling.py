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

def random_sample_anti(n_samples):
    """
    Takes random n_samples from a uniform distribution.
    Returns arrays x and y.
    """
    x = np.array([])
    y = np.array([])
    x_anti = np.array([])
    y_anti = np.array([])

    # Makes arrays of x and y values
    for i in range(n_samples):
        aa = np.random.uniform(0,3.0,4)
        a = random.choice(aa)

        x = np.append(x,a-2)
        x_anti = np.append(x_anti,a-2)
        
        bb = np.random.uniform(0,3.0,4)
        b = random.choice(bb)

        y = np.append(y,b-1.5)
        y_anti = np.append(y_anti,b-1.5)
        
        a_anti = 3.0 - a
        x_anti = np.append(x_anti, a_anti-2)

        b_anti = 3.0 - b
        y_anti = np.append(y_anti, b_anti-1.5)

    return x, y, x_anti, y_anti

def stratified_random_sample(n_samples):
    """
    Takes random n_samples from a linspace.
    Returns arrays x and shuffled y.
    """
    x = np.random.uniform(-2, 1, n_samples).tolist()
    y = np.linspace(-1.5, 1.5, n_samples)

    # np.random.shuffle(y)
    # np.random.shuffle(x)

    x = np.array(x)
    y = np.array(y)

    return x, y