import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def assignment_2():
    data = pd.read_csv('MMC_values.csv') 
    df = pd.DataFrame(data) 


    server_1 = df.loc[df['Servers'] == '1 server(s)']["Values"]
    servers_2 = df.loc[df['Servers'] == '2 server(s)']["Values"]
    servers_4 = df.loc[df['Servers'] == '4 server(s)']["Values"]

    sns.displot(df.loc[df['Servers'] == '1 server(s)'], x="Values", bins=6)

    plt.show()

    print(stats.shapiro(server_1))
    print(stats.shapiro(servers_2))
    print(stats.shapiro(servers_4))

    print("\n")
    fvalue, pvalue = stats.f_oneway(server_1, servers_2, servers_4)

    print(fvalue, pvalue)

    Post_hoc = sp.posthoc_ttest(df, val_col='Values', group_col='Servers', p_adjust='holm')

    print(Post_hoc)

    


    plt.style.use('ggplot')
    ax = sns.boxplot(x="Servers", y="Values", data=data)
    plt.ylabel("Waiting time")


    # Kruskal analysis, not normal distributed
    print("\n", stats.kruskal(server_1, servers_2, servers_4))
    Post_hoc_con = sp.posthoc_conover(df, val_col='Values', group_col='Servers', p_adjust='holm')
    print(Post_hoc_con)
    # plt.show()


def assignment_3():
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
    plt.ylabel("Waiting time")

    plt.show()
assignment_2()
# assignment_3()