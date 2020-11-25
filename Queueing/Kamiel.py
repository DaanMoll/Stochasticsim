"""
Bank renege example

Covers:

- Resources: Resource
- Condition events

Scenario:
  A counter with a random service time and customers who renege. Based on the
  program bank08.py from TheBank tutorial of SimPy 2. (KGM)

"""
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
        NEW_CUSTOMERS = 1100  # Total number of customers
        INTERVAL_CUSTOMERS = 10  # Generate new customers roughly every x seconds


        def source(env, number, interval, counter):
            """Source generates customers randomly"""
            for i in range(number):
                c = customer(env, 'Customer%02d' % i, counter, i, time_in_bank=9.0)
                env.process(c)
                t = random.expovariate(1/interval)
                yield env.timeout(t)


        def customer(env, name, counter, i, time_in_bank):
            """Customer arrives, is served and leaves."""
            arrive = env.now
            # print('%7.4f %s: Here I am' % (arrive, name))

            with counter.request() as req:

                # Wait for the counter
                yield req
                wait = env.now - arrive
                if i > 100:
                    waiting_time.append(wait)
                
                # We got to the counter
                # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

                tib = random.expovariate(1/time_in_bank)
                yield env.timeout(tib)
                # print('%7.4f %s: Finished' % (env.now, name))


        # Setup and start the simulation
        # print('Bank renege')
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

df

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
print(c_values, c_group)