import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis
import pandas as pd
import seaborn as sns

from mandelbrotset.mandelbrot_area import compute_area_mandelbrot
from mandelbrotset.mandelbrot_image import compute_image_mandelbrot

if __name__=="__main__":
    N_max = [value for value in range(10, 101, 10)]
    major = [value for value in range(10, 20, 5)]
    methods = ["random", "lhs", "ortho"]
    
    for method in methods:
        p_samples = []
        pandas_y_values = []
        pandas_x_values = []

        for m in major:
            n = m * m
            for N in N_max:
                for i in range(2):
                    ortho_c = compute_area_mandelbrot(N, 2, n, method)
                    p_samples.append(str(n))

                    pandas_x_values.append(N)
                    pandas_y_values.append(ortho_c[2])

        # # Create an array with the colors you want to use
        data = {'N_values':pandas_x_values, 'area':pandas_y_values, 'amount':p_samples} 

        plt.figure()
        sns.set()

        # Create DataFrame 
        df = pd.DataFrame(data) 

        svm = sns.lineplot(data=data, x="N_values", y="area", hue="amount").set_title(f"{method}")
        figure = svm.get_figure()
        figure.savefig(f'Images/{method}.png', dpi=400)

    
    # N_max = [value for value in range(10, 101, 10)]
    # major = [value for value in range(10, 20, 5)]
    # methods = ["random", "lhs", "ortho"]

    # for method in methods:
    #     for m in major:
    #         n = m * m
    #         areas = []
    #         for N in N_max:
    #             mean = []
    #             for _ in range(10):
    #                 result = compute_area_mandelbrot(N, 5, n, method)
    #                 mean.append(result[2])
    #             areas.append(np.mean(mean))
    #         plt.plot(N_max, areas, label=f"{n} samples, {method}")
    #     plt.legend()
    #     plt.show()

def N_maxtest():    
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
        # print("mean N:", N, np.mean(r_mean),np.mean(lhs_mean), np.mean(ortho_mean))

    r_diff = [abs(area-ortho_areas[-1]) for area in r_areas]
    lhs_diff = [abs(area-lhs_areas[-1]) for area in lhs_areas]
    ortho_diff = [abs(area-ortho_areas[-1]) for area in ortho_areas]

    plt.plot(N_max, r_diff, label=f"Rand")
    plt.plot(N_max, lhs_diff, label=f"Lhs")
    plt.plot(N_max, ortho_diff, label=f"Ortho")

    # print(r_areas[-1], lhs_areas[-1], ortho_areas[-1])
    
    plt.legend()
    plt.show()


    








   