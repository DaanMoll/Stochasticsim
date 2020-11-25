import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def assignment_1():
    data = pd.read_csv('values.csv') 
    df = pd.DataFrame(data) 


    server_1 = df.loc[df['Servers'] == '1 server(s)']["Values"]
    servers_2 = df.loc[df['Servers'] == '2 server(s)']["Values"]
    servers_4 = df.loc[df['Servers'] == '4 server(s)']["Values"]

    print(stats.shapiro(server_1))
    print(stats.shapiro(servers_2))
    print(stats.shapiro(servers_4))

    print("\n")
    fvalue, pvalue = stats.f_oneway(server_1, servers_2, servers_4)

    print(fvalue, pvalue)

    Post_hoc = sp.posthoc_ttest(df, val_col='Values', group_col='Servers', p_adjust='holm')

    print(Post_hoc)

    print("\n", stats.ttest_ind(server_1,servers_2))


    plt.style.use('ggplot')
    ax = sns.boxplot(x="Servers", y="Values", data=data)
    plt.ylabel("Waiting time")

    plt.show()

assignment_1()