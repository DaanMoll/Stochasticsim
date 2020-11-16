import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

def ortho_sample(x, y, n_samples):
    """
    Uses the Ortho-package from canvas. Takes random n_samples with the orthogonal sampling method.
    Returns array x and y.
    """
    counter = 0
    major = n_samples ** 0.5
    integer = np.random.randint(0, 10)

    with open(f"ortho-pack/{int(major)}_{integer}.txt") as coordinates:
        for row in coordinates:
            row = row.split(",")
            row.pop(-1)

            if counter == 0:
                for coordinate in row:
                    x = np.append(x,float(coordinate))
            elif counter == 1:
                for coordinate in row:
                    y = np.append(y,float(coordinate))

            counter += 1
    
    return x, y