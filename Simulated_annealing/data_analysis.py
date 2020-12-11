import pandas as pd
import numpy as np
import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from matplotlib import rcParams

plt.style.use('ggplot')

def plot(route, version):    
    cities = {}
    tsp_file = "a280"

    # Reads from data file and plots and saves all coordinates from cities
    with open(f"TSP_data/{tsp_file}.tsp.txt","r") as reader:
        counter = 0
        for line in reader:
            line = line.strip("\n")
            line = line.split()
    
            if line[0].isdigit():
                # counter+=1
                plt.plot(int(line[1]), int(line[2]), '.')
                # plt.annotate(counter, [int(line[1]), int(line[2])], fontsize=8)

                cities[counter] = (int(line[1]), int(line[2]))
                counter+=1

    for i in range(len(route) - 1):
        city1 = cities[route[i]]
        city2 = cities[route[i+1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    city1 = cities[route[-1]]
    city2 = cities[route[0]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    plt.savefig(f'{version}_route.png')

def convergence_compare():
    files = ["Log_3","Exponential_3"] 
    frames = []
    for file_ in files:
        data = pd.read_csv(f'data/values_{file_}_iter.csv') 
        df = pd.DataFrame(data)
        df =  df.loc[df['Percentage'] == 80] 
        frames.append(df)

    result = pd.concat(frames)

    # Checks if all schedules ar normally distibuted for data with marov chain length
    mcl  = 100
    ndd = df.result[result['Markov'] == mcl] 
    ax = sns.histplot(ndd, x="Cost", hue="Cooling_schedule", kde=True)
    figure = ax.get_figure()
    figure.savefig(f'images/Distributions.png')
    # plt.show()

    # Checks for convergence 
    ax = sns.lineplot(data=result, x="Markov", y="Cost", hue="Cooling_schedule")
    figure = ax.get_figure()
    figure.savefig(f'images/Convergence.png')
    # plt.show()

    # Compares cooling schedules
    ax = sns.boxplot(x="Cooling_schedule", y="Cost", data=result)
    figure = ax.get_figure()
    figure.savefig(f'images/Schedule_compare.png')
    # plt.show()
    
    # Looks for best solutions
    df = result['Cost'] 
    minimal_cost = df.min()
    index = result.loc[df == minimal_cost].index[0]
    old_route = result.get_value(index, 'Init routes') 
    new_route = result.get_value(index, 'Routes') 

    plot(old_route, "Old")
    plot(new_route, "New")


if __name__ == "__main__":
    convergence_compare()
