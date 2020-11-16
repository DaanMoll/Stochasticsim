import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

def random_sample(n_samples):
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
        a = np.random.uniform(0,3.0)
        a_anti = 3.0 - a
        x = np.append(x,a-2)
        x_anti = np.append(x_anti, a_anti-2)

        b = np.random.uniform(0,3.0)
        b_anti = 3.0 - b
        y = np.append(y,b-1.5)
        y_anti = np.append(y_anti, b_anti-1.5)

    return x, y, x_anti, y_anti