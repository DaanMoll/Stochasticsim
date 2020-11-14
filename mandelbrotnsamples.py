import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis

if __name__=="__main__":
    # compute_image_mandelbrot(100, 5, 401, 601)

    N_max = [10, 100, 200, 300, 500]
    n_samples = [50, 100, 200]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for n in n_samples:
        r_areas = []
        lhs_areas = []

        print(n)

        for N in N_max:
            r_mean = []
            lhs_mean = []
            for _ in range(10):
                random = compute_area_mandelbrot(N, 5, n, "random")
                lhs = compute_area_mandelbrot(N, 5, n, "lhs")
                
                r_mean.append(random[2])
                lhs_mean.append(lhs[2])
            
            r_areas.append(np.mean(r_mean))
            lhs_areas.append(np.mean(lhs_mean))

        ax1.plot(N_max, r_areas, label=f"{n} samples rand")
        ax2.plot(N_max, lhs_areas, label=f"{n} samples, lhs")
    
    ax1.legend()
    ax2.legend()
    plt.show()