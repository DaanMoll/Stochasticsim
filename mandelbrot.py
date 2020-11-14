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

def ortho_sample(x, y, n_samples):
    """
    Takes random n_samples with the orthogonal sampling method.
    Returns array x and y.
    """
    print("hallo")
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
    print(x,y)
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

    if kind == "random":
        sample = random_sample(x, y, n_samples)
    elif kind == "lhs":
        sample = lhs_sample(x, y, n_samples)
    elif kind == "ortho":
        sample = ortho_sample(x, y, n_samples)
    
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


if __name__=="__main__":
    # zorgen dat n_samples.txt in ortho-pack staat voor ortho sampling
    # N_max = [value for value in range(10, 101, 10)]
    # N_max.extend([1200, 1400, 1600, 1800, 2000])
    N_max = [10, 20, 50, 100, 200, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    major = 10
    n = major * major

    r_areas = []
    lhs_areas = []
    ortho_areas= []

    for N in N_max:
        r_mean = []
        lhs_mean = []
        ortho_mean = []

        for _ in range(10):
            random = compute_area_mandelbrot(N, 5, n, "random")
            lhs = compute_area_mandelbrot(N, 5, n, "lhs")
            ortho = compute_area_mandelbrot(N, 5, n, "ortho")
            
            r_mean.append(random[2])
            lhs_mean.append(lhs[2])
            ortho_mean.append(ortho[2])
        
        r_areas.append(np.mean(r_mean))
        lhs_areas.append(np.mean(lhs_mean))
        ortho_areas.append(np.mean(ortho_mean))
        print("mean N:", N, np.mean(r_mean),np.mean(lhs_mean), np.mean(ortho_mean))

    r_diff = [abs(area-ortho_areas[-1]) for area in r_areas]
    lhs_diff = [abs(area-lhs_areas[-1]) for area in lhs_areas]
    ortho_diff = [abs(area-ortho_areas[-1]) for area in ortho_areas]

    plt.plot(N_max, r_diff, label=f"Rand")
    plt.plot(N_max, lhs_diff, label=f"Lhs")
    plt.plot(N_max, ortho_diff, label=f"Ortho")

    print(r_areas[-1], lhs_areas[-1], ortho_areas[-1])
    
    plt.legend()
    plt.show()


    








   