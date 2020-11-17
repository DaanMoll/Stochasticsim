import numpy as np
import matplotlib.pyplot as plt
import lhsmdu
from skopt.sampler import Lhs
from skopt.space import Space
from numpy import newaxis
import pandas as pd
import seaborn as sns
import random

from mandelbrotset.mandelbrot_area import compute_area_mandelbrot, compute_area_mandelbrot_anti
from mandelbrotset.mandelbrot_image import compute_image_mandelbrot

def sample_size():
    N = 200
    major = [value for value in range(10, 11, 2)]
    methods = ["Random", "LHS", "Orthogonal"]

    repetitions = 10
    repeat = int(repetitions/2)
    repeat = repetitions

    p_samples = []
    pandas_y_values = []
    pandas_x_values = []

    for method in methods:
        print(method)
        for m in major:
            samples = m * m
            print(samples)
            for i in range(repeat):
                result = compute_area_mandelbrot(N, 2, samples, method)
            
                p_samples.append(method)
                pandas_x_values.append(str(samples))
                pandas_y_values.append(result[2])
        
    # Create an array with the colors you want to use
    data = {'Sample size':pandas_x_values, 'Area':pandas_y_values, 'Sampling method':p_samples} 

    sns.set(font_scale=1.1)

    # Create DataFrame
    df = pd.DataFrame(data) 
    svm = sns.lineplot(data=data, x="Sample size", y="Area", hue="Sampling method", color=sns.set_palette("Set2"))
    svm.set_title(f"Convergence of the estimated area with increased sample size")

    plt.xticks(rotation=45)
    plt.tight_layout()

    figure = svm.get_figure()
    figure.savefig(f'images/samplesize/Ssize{major[0]}-{major[-1]}.png')

def sample_size_anti():
    N = 1000
    major = [value for value in range(5, 6, 2)]
    methods = ["Random"]

    repetitions = 100
    repeat = repetitions

    # p_samples = [] 
    # p_anti_samples = []
    # pandas_y_values = []
    # pandas_y_anti_values = []
    # pandas_x_values = []
    # pandas_x_anti_values = []
    
    var = []
    var_anti = []

    for method in methods:
        print(method)
        for m in major:
            samples = m * m
            print(samples)
            var1 = []
            var2_anti = []

            for i in range(repeat):
                result = compute_area_mandelbrot(N, 2, samples, method)
                # p_samples.append(method)
                # pandas_x_values.append(str(samples))
                # pandas_y_values.append(result[2])
                var1.append(result[2])

                result = compute_area_mandelbrot(N, 2, samples, "Random2")
                # p_anti_samples.append(str(method))
                # pandas_x_anti_values.append(str(samples))
                # pandas_y_anti_values.append(result[2])
                var2_anti.append(result[2])
                    
            var.append(var1)
            var_anti.append(var2_anti)
    
    print(var)
    print("\n\n")
    print(var_anti)

    for i in range(len(var)):
        variation_normal = np.std(var[i])
        variation_anti = np.std(var_anti[i])
        print(major[i]**2)
        print("var normal:", variation_normal)
        print("var anti:", variation_anti)
        print("var normal - anti = ", variation_normal - variation_anti)
        print("If value is positive than anti variance is smaller")
        
    # # Create an array with the colors you want to use
    # data = {'Sample size':pandas_x_values, 'Area':pandas_y_values, 'Sampling method':p_samples} 

    # sns.set(font_scale=1.1)

    # # Create DataFrame
    # df = pd.DataFrame(data) 
    # svm = sns.lineplot(data=data, x="Sample size", y="Area", hue="Sampling method", color=sns.set_palette("Set2"))
    # svm.set_title(f"Convergence of the estimated area with increased sample size")

    # plt.xticks(rotation=45)
    # plt.tight_layout()

    # figure = svm.get_figure()
    # figure.savefig(f'images/samplesize/Ssize{major[0]}-{major[-1]}.png')


def N_max_test():
    N_max = [value for value in range(100, 2001, 50)]
    major = [value for value in range(10, 26, 5)]
    methods = ["Random", "LHS", "Orthogonal"]
    
    for method in methods:
        print(method)
        p_samples = []
        pandas_y_values = []
        pandas_x_values = []
        delta_area = []
        delta_x = []
        delta_samples = []
        
        for m in major:
            samples = m * m
            mean = []
            
            for N in N_max:
                values = []
                for i in range(2):
                    result = compute_area_mandelbrot(N, 2, samples, method, False)
                    
                    p_samples.append(str(samples))
                    pandas_x_values.append(N)
                    pandas_y_values.append(result[2])

                    values.append(result[2])

                mean.append(np.mean(values))
                delta_x.append(N)
                delta_samples.append(str(samples))

            delta_x.pop(-1)
            delta_samples.pop(-1)
            for i in range(0, len(mean) - 1):
                delta_area.append(abs(mean[i] - mean[i+1]))

        # Create an array with the colors you want to use
        data = {'Iterations':pandas_x_values, 'Area':pandas_y_values, 'Sample size':p_samples} 

        sns.set(font_scale=1.1)
        plt.figure()

        # Create DataFrame 
        df = pd.DataFrame(data) 

        svm = sns.lineplot(data=data, x="Iterations", y="Area", hue="Sample size").set_title(f"Sampling method: {method}")
        figure = svm.get_figure()
        figure.savefig(f'images/{method}Nmax{N_max[-1]}.png')

        # Delta plot
        data = {'Iterations':delta_x, 'Delta area':delta_area, 'Sample size':delta_samples} 

        plt.figure()
        plt.xticks(fontsize=12)

        # Create DataFrame 
        df = pd.DataFrame(data) 

        svm = sns.lineplot(data=data, x="Iterations", y="Delta area", hue="Sample size").set_title(f"Sampling method: {method}")
        figure = svm.get_figure()
        figure.savefig(f'images/{method}deltaNmax{N_max[-1]}.png')

if __name__=="__main__":
    sample_size_anti()
    # N_max_test()