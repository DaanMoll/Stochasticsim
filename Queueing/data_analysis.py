import pandas as pd
import numpy as np
import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from matplotlib import rcParams

plt.style.use('ggplot')
queueing_type = "Longtail"

def comparing_servers():
    data = pd.read_csv(f'{queueing_type}_values.csv') 
    df = pd.DataFrame(data) 
    data1 = pd.read_csv(f'Waiting_data/MMC_values2.csv') 
    df1 =  pd.DataFrame(data1)
    data2 = pd.read_csv(f'Waiting_data/MDC_values2.csv') 
    df2 =  pd.DataFrame(data2)

    server_1a =  df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of customers'] == 100000)]
    servers_2a =  df.loc[(df['Servers'] == '2 server(s)') & (df['Amount of customers'] == 100000)]
    servers_4a =  df.loc[(df['Servers'] == '4 server(s)') & (df['Amount of customers'] == 100000)]
    server_1b =  df1.loc[(df1['Servers'] == '1 server(s)') & (df1['Amount of customers'] == 100000)]
    servers_2b =  df1.loc[(df1['Servers'] == '2 server(s)') & (df1['Amount of customers'] == 100000)]
    servers_4b =  df1.loc[(df1['Servers'] == '4 server(s)') & (df1['Amount of customers'] == 100000)]
    server_1c =  df2.loc[(df2['Servers'] == '1 server(s)') & (df2['Amount of customers'] == 100000)]
    servers_2c =  df2.loc[(df2['Servers'] == '2 server(s)') & (df2['Amount of customers'] == 100000)]
    servers_4c =  df2.loc[(df2['Servers'] == '4 server(s)') & (df2['Amount of customers'] == 100000)]
    
    l_a = [server_1a, servers_2a, servers_4a]
    l_b = [server_1b, servers_2b, servers_4b]
    l_c = [server_1c, servers_2c, servers_4c]
    
    counter = 0
    for a in l_a:
        for b in l_b:
            for c in l_c:
                counter +=1
                print(stats.ttest_ind(a["Values"], b["Values"], equal_var = True))
                print(stats.ttest_ind(b["Values"], c["Values"], equal_var = True))
                print(stats.ttest_ind(a["Values"], c["Values"], equal_var = True))
                print("\n",counter,"\n")

    frames = [server_1, servers_2, servers_4]
    result = pd.concat(frames)
    
    # print(stats.shapiro(server_1["Values"]))
    # print(stats.shapiro(servers_2["Values"]))
    # print(stats.shapiro(servers_4["Values"]))
    # sns.set(font_scale=1.25)
    # ax = sns.histplot(result, x="Values", hue="Servers", kde=True)
    
    # plt.title("M/M/C distributions for different servers")
    # plt.xlabel("Mean waiting time")
    # print(np.mean(server_1["Values"]))
    # print(np.std(server_1["Values"]))
    # print(np.mean(servers_2["Values"]))
    # print(np.std(servers_2["Values"]))
    # print(np.mean(servers_4["Values"]))
    # print(np.std(servers_4["Values"]))
    # figure = ax.get_figure()
    
    # figure.savefig(f'images/{queueing_type}_Distributions.png', bbox_inches='tight' )
  
    # plt.show()

    # # print(stats.ttest_ind(server_1["Values"], servers_2["Values"], equal_var = True))
    # # print(stats.ttest_ind(server_1["Values"], servers_4["Values"], equal_var = True))
    # # print(stats.ttest_ind(server_2["Values"], servers_4["Values"], equal_var = True))

    sns.set(font_scale=1.25)
    b = sns.boxplot(x="Servers", y="Values", data=data)
    b.set_title(queueing_type)
    plt.ylabel("Waiting time")
    plt.xlabel(" ")

    plt.title("Comparing M/D/C queues")
    figure = b.get_figure()
    figure.savefig(f'images/{queueing_type}1_Boxplot_comp.png')
    # plt.show()
    
def rho_measures():
    """
    In this function various standard deviations of rho values will
    be compared with eachother based on different measurements.
    Furthermore a histogram is plotted gor rho = 0.9 
    """
    data = pd.read_csv(f'MM_values.csv') 
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
            rho =  df.loc[(df['Rho'] == f' Value: {a}') & (df['Amount of customers'] == value)]["Values"]
            std_rho = np.std(rho)
            dicti[a].append(std_rho)

        dicti[a] = scipy.interpolate.make_interp_spline(values, dicti[a])
        dicti[a] = dicti[a](x)
        plt.semilogy(x, dicti[a], label=f"\u03C1 = {a}")
        plt.legend(fontsize=12)

    plt.xlabel("Measurements", fontsize=14)
   
    plt.xticks(values)
    plt.ylabel("Std of Average waiting time", fontsize=14)
    plt.title("Comparison of different \u03C1 value", fontsize=16)
    plt.savefig('various_rho.png')

    plt.show()

    # Takes all the data for Rho == 0.9
    a =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 25000)]
    b =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 50000)]
    c =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 75000)]
    d =  df.loc[(df['Rho'] ==  ' Value: 0.9') & (df['Amount of customers'] == 100000)]

    # Makes new frame for rho == 0.9
    frames = [a, b, c, d]
    result = pd.concat(frames)
    
    sns.set(font_scale=1.25)
    
    # Plots histogram for rho == 0.9
    a = sns.histplot(result, x="Values", hue="Amount of customers", kde=True)
    
    plt.title("Distributions for \u03C1 = 0.9")
    
    plt.xlabel("Mean waiting time")
    # savefig(f'images/rho=0.9_Distributions.png')
    
    figure = a.get_figure()
    figure.savefig(f'images/rho=0.9_Distributions.png', bbox_inches='tight')
    plt.show()


def comparing_SJF():
    data = pd.read_csv('MMC_values.csv') 
    df = pd.DataFrame(data) 

    data1 = pd.read_csv('SJF_values.csv')
    df1 = pd.DataFrame(data1) 

    server_1 =df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of customers'] == 100000)]
    server_SJF = df1.loc[df1["Servers"] =='SJF']

    frames = [server_1, server_SJF]
    result = pd.concat(frames)


    print(stats.shapiro(server_1["Values"]))
    print(stats.shapiro(server_SJF["Values"]))
    print(np.mean(server_1["Values"]))
    print(np.std(server_1["Values"]))
    print(np.mean(server_SJF["Values"]))
    print(np.std(server_SJF["Values"]))

    print("\n", stats.ttest_ind(server_1["Values"],server_SJF["Values"]))
    sns.set(font_scale=1.25)
    ax = sns.boxplot(x="Servers", y="Values", data=result)
    # ax.set_title("Longtail")
    plt.xlabel("")
    plt.ylabel("Waiting time")
    plt.xlabel(" ")
    
    plt.title("Comparing SJF to FIFO ")

    figure = ax.get_figure()
    figure.savefig(f'images/SJF_Boxplot_comp.png')
    plt.show()

# a()
# rho_measures()
# comparing_servers()
comparing_SJF()
