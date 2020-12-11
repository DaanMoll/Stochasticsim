import pandas as pd
import numpy as np
import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from matplotlib import rcParams
from plot import plot

plt.style.use('ggplot')

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
