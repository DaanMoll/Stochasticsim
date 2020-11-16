import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

def lhs_sample(n_samples):
    """
    Takes random n_samples with the lhs method.
    Returns array x and y.
    """
    x = np.array([])
    y = np.array([])
    x_anti = np.array([])
    y_anti = np.array([])

    # Makes the space of points which van be chosen from
    space = Space([(-2.,1.), (-1.5, 1.5)])  

    # Chooses which kind oh lhs will be used
    lhs = Lhs(lhs_type="classic", criterion=None)

    # Generates n_samples withhi the chosen space
    coordinates = lhs.generate(space.dimensions, n_samples)

    # appends all x and y values to array
    for coordinate in coordinates:
        a = coordinate[0]
        x = np.append(x,a)

        a_anti = 3.0 - (a + 2.0)
        x_anti = np.append(x_anti, a_anti) 

        b = coordinate[1]
        y = np.append(y,b)

        b_anti = 3.0 - (b + 1.5)
        y_anti = np.append(y_anti, b_anti) 

    return x, y, x_anti, y_anti