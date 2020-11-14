import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

def random_sample(x, y, n_samples):
    """
    Takes random n_samples from a uniform distribution.
    Returns array x and y.
    """

    # Makes arrays of x and y values
    for i in range(n_samples):
        a = np.random.uniform(-2,1)
        x = np.append(x,a)
        b = np.random.uniform(-1.5,1.5)
        y = np.append(y,b)

    return x, y