import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random
import math
import pandas as pd
import sys
import matplotlib.pyplot as plt

def cooling_schedule(start_temp, max_iterations, iteration, kind):
    if kind  == "Linear":
        # multiplicative
        alpha = 1
        current_temp = start_temp/(1 + alpha*iteration)
        # additive
        # current_temp = start_temp * ((max_iterations-iteration) / max_iterations)
    elif kind == "Log":
        alpha = 50
        # multiplicative
        current_temp =  start_temp/(alpha * (math.log(iteration + 1, 10)))
    elif kind == "Exponential":
        # additive 
        # unit = 1 + np.exp((2 * np.log(start_temp)/max_iterations) * (iteration - (0.5 * max_iterations)))
        # current_temp = start_temp * (1 / unit)
        # multiplicative
        current_temp = start_temp*0.9**iteration
    elif kind == "Quadratic":
        # additive
        # current_temp = start_temp * ((max_iterations - iteration)/max_iterations)**2

        # multiplicative
        alpha = 1
        current_temp =  start_temp/(1 + alpha * iteration**2)
    
    return current_temp

if __name__ == '__main__':
    # cooling = sys.argv[1]

    temperature = {}
    temperature["Exponential"] = 200
    temperature["Linear"] = 200
    temperature["Log"] = 3000
    temperature["Quadratic"] = 200

    cooling_schedules = ["Linear","Log", "Exponential", "Quadratic"]

    for cooling in cooling_schedules:
        temps = []
        # start_temp = temperature[cooling]
        start_temp = 1000
        print(start_temp)

        max_iterations = 40
        for iteration in range(1, max_iterations):
            current_temp = cooling_schedule(start_temp, max_iterations, iteration, cooling)
            temps.append(current_temp)
    
        print(cooling)
        plt.plot(range(1, max_iterations), temps, label=cooling)
        
    plt.legend(fontsize=14)
    plt.ylabel("Temperature", fontsize=14)
    plt.xlabel("Iteration", fontsize=14)
    plt.title("Temperature decay for different cooling schedules", fontsize=16)
    plt.savefig("images/CoolingSchedules")
    # plt.show()
    