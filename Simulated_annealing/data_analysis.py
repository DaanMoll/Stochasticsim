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
    """
    Creates boxplots and convergence comparison between the 4 cooling systems
    """
    files = ["Log_301", "Exponential_301", "Linear_301", "Quadratic_301"] 
    frames = []
    for file_ in files:
        data = pd.read_csv(f'data/values_{file_}_iter.csv') 
        df = pd.DataFrame(data)
        df = df.loc[df['Percentage'] == 80] 
        frames.append(df)

    result = pd.concat(frames)

    # Checks if all schedules ar normally distibuted for data with marov chain length
    mcl  = 100
    ndd = result.loc[result['Markov'] == mcl] 
    ax = sns.histplot(ndd, x="Cost", hue="Cooling_schedule", kde=True)
    figure = ax.get_figure()
    figure.savefig(f'images/Distributions.png')
    plt.show()

    # Checks for convergence 
    ax = sns.lineplot(data=result, x="Markov", y="Cost", hue="Cooling_schedule")
    figure = ax.get_figure()
    figure.savefig(f'images/Convergence.png')
    plt.show()

    # Compares cooling schedules
    ax = sns.boxplot(x="Cooling_schedule", y="Cost", data=result)
    figure = ax.get_figure()
    figure.savefig(f'images/Schedule_compare.png')
    plt.show()
    
    # Looks for best solutions
    df = result['Cost'] 
    minimal_cost = df.min()
    print(f"best score = {minimal_cost}")
    index = result.loc[df == minimal_cost].index[0]
    old_route = result._get_value(index, 'Init routes') 
    new_route = result._get_value(index, 'Routes') 

    old_route  = old_route.split(", ")
    old_route[0] = old_route[0][1:]
    old_route[-1] = old_route[-1][:-1]

    new_route  = new_route.split(", ")
    new_route[0] = new_route[0][1:]
    new_route[-1] = new_route[-1][:-1]
    print(old_route)
    print(new_route)
    plot(old_route, "Old")
    plot(new_route, "New")
         
def conv_lin_ml():
    """
    Creates plot of convergence of the linear cooling schedule at different markov lengts
    """
    file_ = "Linear_100"
    frames = []
    costs1 = []

    
    data = pd.read_csv(f'data/Convergence_{file_}.csv') 
    
    df = pd.DataFrame(data)
    markov_lengths = [10, 25, 50, 75, 100, 125, 150]
    frames.append(df)

    result = pd.concat(frames)

    all_iters = []

    itera = result["Iter2"]
    for iters in itera:
        iters = iters[1:]
        iters = iters[:-1]
        iters = iters.split(", ")
        all_iters.append(iters)
    
    print(all_iters[0][-1], all_iters[0][-2], len(all_iters[0]))

    cost_run = result["Cost in run"]
    for costs in cost_run:
        cost = costs[1:]
        cost = cost[:-1]
        cost = cost.split(", ")
        costs1.append(cost)

    print(len(costs1[0]), costs1[0][-1], len(costs1))

    markovs = []
    total_costs = []
    iterations = []

    for i in range(len(costs1)):
        ml = markov_lengths[i%7]
        run = costs1[i]
        for j in range(len(costs1[0])):
            iterations.append(j+1)
            markovs.append(ml)
            total_costs.append(float(run[j]))

    print(iterations[-1])

    data = {"Markov":markovs, "Cost in run":total_costs, "Iter2":iterations}
    df = pd.DataFrame(data)

    ax = sns.lineplot(data=df, x="Iter2", y="Cost in run", hue="Markov", ci=None)
    
    ax.set_title(f"Convergence of the linear cooling schedule with different ML")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set(xscale="linear", yscale="log")
    figure = ax.get_figure()
    figure.savefig(f'images/Convergence_lin.png')

def convergence_all():
    """
    Creates png of convergence of the cooling schedules over iterations
    """
    files = ["Log_301", "Exponential_301", "Linear_301", "Quadratic_301"]
    frames = []

    for file_ in files:
        data = pd.read_csv(f'/Users/daan/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master Computational Science/Stochastic/data2/values_{file_}_iter100ml.csv') 
        df = pd.DataFrame(data)
        frames.append(df)

    result = pd.concat(frames)
    
    ax = sns.lineplot(data=result, x="Iter2", y="Cost in run", hue="Cooling_schedule", ci=None)
    ax.set_title(f"Convergence of cost using different cooling schedules")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set(xscale="log", yscale="log")

    figure = ax.get_figure()
    figure.savefig(f'images/Convergence_all4.png')


if __name__ == "__main__":
    # convergence_compare()
    convergence_all()
    conv_lin_ml()
