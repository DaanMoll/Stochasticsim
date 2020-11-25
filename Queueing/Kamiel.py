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
    for _ in range(1000):
        waiting_time = []
        RANDOM_SEED = random.randint(1, 600)
        NEW_CUSTOMERS = 1100  # Total number of customers
        INTERVAL_CUSTOMERS = 10  # Generate new customers roughly every x seconds


        def source(env, number, interval, counter):
            """Source generates customers randomly"""
            for i in range(number):
                c = customer(env, 'Customer%02d' % i, counter, i, time_in_bank=5.0)
                env.process(c)
                t = random.expovariate(1/interval)
                yield env.timeout(t)


        def customer(env, name, counter, i, time_in_bank):
            """Customer arrives, is served and leaves."""
            arrive = env.now
            tib = random.expovariate(1/time_in_bank)
            # print('%7.4f %s: Here I am' % (arrive, name))
            
            with counter.request() as req:

                # Wait for the counter
                yield req
                wait = env.now - arrive
                if i > 100:
                    waiting_time.append(wait)
                
                

                # We got to the counter
                # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

                
                yield env.timeout(tib)
                # print('%7.4f %s: Finished' % (env.now, name))

        random.seed(RANDOM_SEED)
        env = simpy.Environment()

        # Start processes and run
        counter = simpy.resources.resource.Resource(env, capacity=C)
        env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/C, counter))
        env.run()

        c_values.append(np.mean(waiting_time))
        c_group.append(f"{C} server(s)")

data = {'Servers':c_group, "Values":c_values}
df = pd.DataFrame(data) 
df
df.to_csv("values.csv")

