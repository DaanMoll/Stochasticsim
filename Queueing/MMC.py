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
customers = []
cs = [1,2,4]
NEW_CUSTOMERSS = [100000]

for C in cs:
    for NEW_CUSTOMERS in NEW_CUSTOMERSS:    
        for _ in range(500):
            waiting_time = []
            RANDOM_SEED = random.randint(1,100000000)
            INTERVAL_CUSTOMERS = 10

            def source(env, number, interval, counter):
                """Source generates customers randomly"""
                for i in range(number):
                    c = customer(env, 'Customer%02d' % i, counter, i, job_time=9)
                    env.process(c)
                    t = random.expovariate(1/interval)
                    yield env.timeout(t)

            def customer(env, name, counter, i, job_time):
                """Customer arrives, is served and leaves."""
                arrive = env.now
                tib = random.expovariate(1/job_time)
                # print('%7.4f %s: Here I am' % (arrive, name))
                
                with counter.request() as req:
                    # Wait for the counter
                    yield req
                    wait = env.now - arrive
                    # if i > 100:
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

            customers.append(f"{NEW_CUSTOMERS}")
            c_values.append(np.mean(waiting_time))
            c_group.append(f"{C} server(s)")
            if _%10 ==0:
                print(f"servers: {C}, simulation: {_}, customers: {NEW_CUSTOMERS}")
data = {'Servers':c_group, "Values":c_values, "Amount of Customers":customers}
df = pd.DataFrame(data) 
df
df.to_csv("MMC_values2.csv")

print("Done")