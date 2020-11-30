import pandas as pd
import numpy as np
import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from matplotlib import rcParams

plt.style.use('ggplot')
queueing_type = "MMC"

def comparing_servers():
    data = pd.read_csv(f'{queueing_type}_values2.csv') 
    df = pd.DataFrame(data) 

    server_1 =  df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of customers'] == 100000)]
    servers_2 =  df.loc[(df['Servers'] == '2 server(s)') & (df['Amount of customers'] == 100000)]
    servers_4 =  df.loc[(df['Servers'] == '4 server(s)') & (df['Amount of customers'] == 100000)]

    
    print(server_1)
    frames = [server_1, servers_2, servers_4]
    result = pd.concat(frames)
    
    print(stats.shapiro(server_1["Values"]))
    print(stats.shapiro(servers_2["Values"]))
    print(stats.shapiro(servers_4["Values"]))

    ax = sns.displot(result, x="Values", hue="Servers", kde=True)
    plt.title(f"{queueing_type} distributions for different servers")
    ax.savefig(f'images/{queueing_type}_Distributions.png')
  
    plt.show()

  

    print("\n")
    fvalue, pvalue = stats.f_oneway(server_1["Values"], servers_2["Values"], servers_4["Values"])

    # p value lager dan 0.05 dus significant verschil tussen de 3
    print(fvalue, pvalue)

    Post_hoc = sp.posthoc_ttest(df, val_col='Values', group_col='Servers', p_adjust='holm')

    print(Post_hoc)

    # Kruskal analysis, not normal distributed
    print("\n", stats.kruskal(server_1["Values"], servers_2["Values"], servers_4["Values"]))
    Post_hoc_con = sp.posthoc_conover(df, val_col='Values', group_col='Servers', p_adjust='holm')
    print(Post_hoc_con)

    b = sns.boxplot(x="Servers", y="Values", data=data)
    b.set_title(queueing_type)
    plt.ylabel("Waiting time")

    plt.title("Comparing M/M/C queues")
    figure = b.get_figure()
    figure.savefig(f'images/{queueing_type}_Boxplot1_comp.png')
    plt.show()
    
def rho_measures():
    """
    In this function various standard deviations of rho values will
    be compared with eachother based on different measurements.
    Furthermore a histogram is plotted gor rho = 0.9 
    """
    queueing_type = "MM_values2"
    data = pd.read_csv(f'{queueing_type}.csv') 
    df = pd.DataFrame(data)

    # Initializing variables
    dicti = {}
    values = [25000, 50000, 75000, 100000]
    x = np.linspace(25000, 100000, 10000)
    dicti["0.1"] = []
    dicti["0.3"] = []
    dicti["0.6"] = []
    dicti["0.9"] = []
    Rhos = ["0.1", "0.3", "0.6", "0.9"]

    # Puts all the stanard deviations into the dictionary and plots them
    for a in Rhos: 
        for value in values:
            print(a)
            rho =  df.loc[(df['Rho'] == f' Value: {a}') & (df['Amount of customers'] == value)]["Values"]
            std_rho = np.std(rho)
            print(dicti[a])
            dicti[a].append(std_rho)

        dicti[a] = scipy.interpolate.make_interp_spline(values, dicti[a])
        dicti[a] = dicti[a](x)
        plt.semilogy(x, dicti[a], label=f"\u03C1 = {a}")
        plt.legend()

    plt.xlabel("Measurements")
    plt.xticks(values)
    plt.ylabel("Standard deviation")
    plt.title("Standard deviations for different measures")
    # plt.savefig('various_rho.png')
    plt.show()

    # Takes all the data for Rho == 0.9
    a =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 25000)]
    b =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 50000)]
    c =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 75000)]
    d =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 100000)]

    print(stats.shapiro(a["Values"]))
    
    print(stats.shapiro(b["Values"]))
    
    print(stats.shapiro(c["Values"]))
    
    print(stats.shapiro(d["Values"]))
    # Makes new frame for rho == 0.9
    frames = [a, b, c, d]
    result = pd.concat(frames)
    
    # Plots histogram for rho == 0.9
    ax1 = sns.displot(result, x="Values", hue="Amount of customers", kde=True)
    plt.title("Distributions for \u03C1 = 0.9")
    ax1.savefig(f'images/rho=0.9_Distributions.png')
    plt.show()


def comparing_SJF():
    data = pd.read_csv('MMC_values2.csv') 
    df = pd.DataFrame(data) 

    data1 = pd.read_csv('SJF_values2.csv')
    df1 = pd.DataFrame(data1) 

    server_1 =df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of customers'] == 100000)]
    server_SJF = df1.loc[df1["Servers"] =='SJF']

    
    frames = [server_1, server_SJF]
    result = pd.concat(frames)

    print(server_1)
    print(stats.shapiro(server_1["Values"]))
    print(stats.shapiro(server_SJF["Values"]))
    print("\n", stats.ttest_ind(server_1["Values"],server_SJF["Values"]))
    ax = sns.boxplot(x="Servers", y="Values", data=result)
    ax.set_title("Longtail")
    plt.ylabel("Waiting time")
    plt.title("Comparing SJF to normal")

    figure = ax.get_figure()
    figure.savefig(f'images/SJF_Boxplot_comp.png')
    plt.show()




# a()
# rho_measures()

comparing_servers()
# comparing_SJF()