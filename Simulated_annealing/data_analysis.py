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

def costs_plot():
    files = ["Log_301", "Exponential_301", "Linear_301", "Quadratic_301"]
    costs1 = [] 
    total_300 = []
    cooling_schedule = []
    iterations = []

    for file_ in files:
        name = file_.strip("_301")
        frames = []
        
        data = pd.read_csv(f'/Users/daan/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master Computational Science/Stochastic/data/values_{file_}_iter100ml.csv') 
        df = pd.DataFrame(data)
        df = df.loc[df['Percentage'] == 80] 
        frames.append(df)

        result = pd.concat(frames)

        cost_run = result["Cost in run"]
        for costs in cost_run:
            cost = costs[1:]
            cost = cost[:-1]
            cost = cost.split(", ")
            costs1.append(cost)

        for i in range(len(costs1[0])):
            all_300 = []
            for iteration in costs1:
                all_300.append(float(iteration[i]))

            total_300.append(np.mean(all_300))
            cooling_schedule.append(name)
            iterations.append(i+1)
    
    print(iterations[-1], "einde")
    exit()
    data = {"Cost in run":total_300, "Iter2":iterations, "Cooling":cooling_schedule}
    df = pd.DataFrame(data) 
    
    ax = sns.lineplot(data=df, x="Iter2", y="Cost in run", hue="Cooling")
    
    ax.set_title(f"Convergence of cost using different cooling schedules")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set(xscale="linear", yscale="log")
    figure = ax.get_figure()
    figure.savefig(f'images/Convergence100_{name}.png')

def costs_plot2():
    file_ = "Log_301"
    costs1 = [] 
    total_300 = []
    cooling_schedule = []
    iterations = []
    markov = []

    
    data = pd.read_csv(f'/Users/daan/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master Computational Science/Stochastic/data/values_{file_}_iter10ml_2.csv') 
    
    df = pd.DataFrame(data)
    # markov_lengths = [10, 25, 50, 75, 100, 125, 150]
    markov_lengths = [100]
    for ml in markov_lengths:
        frames = []
        ff = df.loc[df['Markov'] == ml] 
        frames.append(ff)

        result = pd.concat(frames)

        cost_run = result["Cost in run"]

        for costs in cost_run:
            cost = costs[1:]
            cost = cost[:-1]
            cost = cost.split(", ")
            costs1.append(cost)

        for i in range(len(costs1[0])):
            all_300 = []
            for iteration in costs1:
                all_300.append(float(iteration[i]))

            total_300.append(np.mean(all_300))
            markov.append(ml)
            iterations.append(i+1)
        
    data = {"Cost in run":total_300, "Iter2":iterations, "Markov":markov}
    df = pd.DataFrame(data) 
    
    ax = sns.lineplot(data=df, x="Iter2", y="Cost in run", hue="Markov")
    
    ax.set_title(f"Convergence of log with different ML")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set(xscale="linear", yscale="log")
    figure = ax.get_figure()
    figure.savefig(f'images/Convergence_ML_LOG.png')

def costs_plot3():
    files = ["Log_301", "Exponential_301", "Linear_301", "Quadratic_301"]
    frames = []

    for file_ in files:
        name = file_.strip("_301")
        
        data = pd.read_csv(f'data/values_{file_}_iter100ml.csv') 
        df = pd.DataFrame(data)
        # df = df.loc[df['Percentage'] == 80] 
        frames.append(df)

    result = pd.concat(frames)
    print("hey kamiel")
    sns.color_palette("tab10")
    ax = sns.lineplot(data=result, x="Iter2", y="Cost in run", hue="Cooling_schedule", ci=None)
    print("nu gaat die lekker")
    sns.color_palette("tab10")
    ax.set_title(f"Convergence of cost using different cooling schedules")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set(xscale="linear", yscale="log")
    figure = ax.get_figure()
    figure.savefig(f'images/Convergence_all4.png')


if __name__ == "__main__":
    # convergence_compare()
    # costs_plot()
    # costs_plot2()
    costs_plot3()