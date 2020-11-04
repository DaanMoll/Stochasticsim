import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
import pynolh
import random
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis
from scipy.stats import ortho_group
import pandas as pd
import seaborn as sns

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

def lhs_sample(x, y, n_samples):
    """
    Takes random n_samples with the lhs method.
    Returns array x and y.
    """

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
    plt.show()


def compute_area_mandelbrot(N_max, some_threshold, n_samples, kind):
    """
    Computes the Area of the mandel brot set with the monte carlo method.
    Takes as initial values N_max(integer), some_threshold(Number), n_samples(integer), kind(string = random, lhs)
    returns the amount of hits and the ratio of hits/total shots
    """

    # Initializes values
    hits = 0
    x = np.array([])
    y = np.array([])

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
    
    # Calculates the ratio hits/total shots
    ratio = hits/n_samples**2

    # Calculates the area
    area = ratio * 9

    return hits, ratio, area


if __name__=="__main__":

    pandas_y_values = []
    pandas_x_values = []
    pandas_sim = []
    random = []
    random_means = []
    lhs= []
    lhs_means = []
    x = []
    
    for a in range(1,20):
        x.append(a*20)
        for i in range(10):
            random_c = compute_area_mandelbrot(100, 5, a*20, "random")
            lhs_c = compute_area_mandelbrot(100, 5, a*20, "lhs")
            # print("Random: ",random_c)
            # print("LHS: ",lhs)
            lhs.append(lhs_c[2])
            random.append(random_c[2])
            pandas_x_values.append(a*20)
            pandas_y_values.append(lhs_c[2])
            pandas_sim.append("lhs")
            pandas_y_values.append(random_c[2])
            pandas_x_values.append(a*20)
            pandas_sim.append("random")
        
        lhs_means.append(np.mean(lhs))
        random_means.append(np.mean(random))
        # print("Random mean = ",np.mean(random))
        # print("Random std = ", np.std(random))
        # print("LHS mean = ",np.mean(lhs))
        # print("LHS std = ",np.std(lhs))

    # plt.plot(x, lhs_means, label = "lhs", alpha= 0.75)
    # plt.plot(x, random_means, label = "Random", alpha = 0.75)
    # plt.rcParams["figure.figsize"] = [20,9]

    print(len(pandas_sim), len(pandas_x_values), len(pandas_y_values))
    data = {'Sim':pandas_sim, 'samples':pandas_x_values, 'area':pandas_y_values} 
    sns.set()
    # Create DataFrame 
    df = pd.DataFrame(data) 
    
    # Print the output. 
    print(df) 
    sns.lineplot(data=data, x="samples", y="area", hue="Sim")
    plt.show()
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax.plot(x, lhs_means, label = "lhs", alpha= 0.5)
    ax.plot(x, random_means, label = "Random", alpha = 0.5)
    ax.legend()
    fig.show()
    fig.savefig('fig1.png', dpi = 300)
    # fig.close()
    