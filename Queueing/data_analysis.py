import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

plt.style.use('ggplot')
queueing_type = "MDC"

def comparing_servers():
    data = pd.read_csv(f'{queueing_type}_values.csv') 
    df = pd.DataFrame(data) 

    server_1 = df.loc[df['Servers'] == '1 server(s)']["Values"]
    servers_2 = df.loc[df['Servers'] == '2 server(s)']["Values"]
    servers_4 = df.loc[df['Servers'] == '4 server(s)']["Values"]

    ax1 = sns.displot(data, x="Values", hue="Servers", kde=True)
    
    plt.xlabel("Waiting time (s)")
    plt.title(queueing_type)

    ax1.savefig(f'images/{queueing_type}_Comp_servers.png')
    # plt.show()
    
    print(stats.shapiro(server_1))
    print(stats.shapiro(servers_2))
    print(stats.shapiro(servers_4))

    print("\n")
    fvalue, pvalue = stats.f_oneway(server_1, servers_2, servers_4)

    # p value lager dan 0.05 dus significant verschil tussen de 3
    print(fvalue, pvalue)

    Post_hoc = sp.posthoc_ttest(df, val_col='Values', group_col='Servers', p_adjust='holm')

    print(Post_hoc)

    # Kruskal analysis, not normal distributed
    print("\n", stats.kruskal(server_1, servers_2, servers_4))
    Post_hoc_con = sp.posthoc_conover(df, val_col='Values', group_col='Servers', p_adjust='holm')
    print(Post_hoc_con)

    b = sns.boxplot(x="Servers", y="Values", data=data)
    b.set_title(queueing_type)
    plt.ylabel("Waiting time (s)")

    # plt.title("Comparing servers")
    figure = b.get_figure()
    figure.savefig(f'images/{queueing_type}_Boxplot_comp.png')
    # plt.show()
    
def rho_measures():
    data = pd.read_csv(f'{queueing_type}_wait_values.csv') 
    df = pd.DataFrame(data)
    ax = sns.displot(data, x="Values", hue="Rho", kde=True)

    Customers_1 = df.loc[df['Rho'] == ' Value: 0.1']["Values"]
    Customers_2 = df.loc[df['Rho'] == ' Value: 0.2']["Values"]
    Customers_3 = df.loc[df['Rho'] == ' Value: 0.25']["Values"]
    plt.xlabel("Waiting time")

    print(stats.shapiro(Customers_10))
    print(stats.shapiro(Customers_5))

    ax.savefig(f'images/{queueing_type}_Rho_measures.png')
    plt.show()

def comparing_SJF():
    data = pd.read_csv('MMC_values.csv') 
    df = pd.DataFrame(data) 

    data1 = pd.read_csv('SJF_values.csv')
    df1 = pd.DataFrame(data1) 

    server_1 = df.loc[df['Servers'] == '1 server(s)']
    server_SJF = df1.loc[df1["Servers"] =='SJF']

    print(server_SJF)
    print(server_1)

    frames = [server_1, server_SJF]
    result = pd.concat(frames)
  
    print("\n", stats.ttest_ind(server_1["Values"],server_SJF["Values"]))
    ax = sns.boxplot(x="Servers", y="Values", data=result)
    # ax.set_title("Longtail")
    plt.ylabel("Waiting time (s)")
    plt.title("M/M/1 with different queueing disciplines")

    figure = ax.get_figure()
    figure.savefig(f'images/SJF_Boxplot_comp.png')
    plt.show()


def a():
    data = pd.read_csv('MMC_values2.csv') 
    df = pd.DataFrame(data) 

    a =  df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of Customers'] == 50000)]["Values"]
    b =  df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of Customers'] == 100000)]["Values"]
    c =  df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of Customers'] == 200000)]["Values"]
    d =  df.loc[(df['Servers'] == '1 server(s)') & (df['Amount of Customers'] == 300000)]["Values"]
    plt.legend(loc='upper left')
    ax1 = sns.displot(data, x="Values", hue="Amount of Customers", kde=True)
    plt.show()

    print(stats.shapiro(a))
    print(stats.shapiro(b))
    print(stats.shapiro(c))
    print(stats.shapiro(d))
    print(a)


# a()
# rho_measures()
comparing_servers()
# comparing_SJF()