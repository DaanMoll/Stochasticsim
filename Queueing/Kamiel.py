import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import numpy as np
import simpy

c_values = []
c_group = []
cs = [1,2,4]

for C in cs:
    for _ in range(100):
        waiting_time = []
        RANDOM_SEED = random.randint(1, 600)
        NEW_CUSTOMERS = 1100
        INTERVAL_CUSTOMERS = 10 

        def source(env, number, interval, counter):
            """Source generates customers randomly"""
            for i in range(number):
                c = customer(env, 'Customer%02d' % i, counter, i, job_time=9.0)
                env.process(c)
                t = random.expovariate(1/interval)
                yield env.timeout(t)

        def customer(env, name, counter, i, job_time):
            """Customer arrives, is served and leaves."""
            arrive = env.now

            with counter.request() as req:
                # Wait for the counter
                yield req
                wait = env.now - arrive
                if i > 100:
                    waiting_time.append(wait)

                jb = random.expovariate(1/job_time)
                yield env.timeout(jb)

        # Setup and start the simulation
        random.seed(RANDOM_SEED)
        env = simpy.Environment()

        # Start processes and run
        counter = simpy.Resource(env, capacity=C)
        env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/C, counter))
        env.run()

        c_values.append(np.mean(waiting_time))
        c_group.append(f"{C} server(s)")
    
data = {'Servers':c_group, "Values":c_values}
df = pd.DataFrame(data) 

server_1 = df.loc[df['Servers'] == '1 server(s)']["Values"]
servers_2 = df.loc[df['Servers'] == '2 server(s)']["Values"]
servers_4 = df.loc[df['Servers'] == '4 server(s)']["Values"]

fvalue, pvalue = stats.f_oneway(server_1, servers_2, servers_4)

print(fvalue, pvalue)

Post_hoc = sp.posthoc_ttest(df, val_col='Values', group_col='Servers', p_adjust='holm')

print(Post_hoc)

plt.style.use('ggplot')
ax = sns.boxplot(x="Servers", y="Values", data=data)
plt.ylabel("Waiting time")

plt.show()
# print(c_values, c_group)