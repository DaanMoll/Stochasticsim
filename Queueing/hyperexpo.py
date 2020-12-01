import random
import simpy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Total number of customers
NEW_CUSTOMERSS = [100000]
# Generate new customers roughly every x seconds
INTERVAL_CUSTOMERS = 10

cs = [1, 2, 4]
c_values = []
c_group = []
Customer = []

for C in cs:
    for NEW_CUSTOMERS in NEW_CUSTOMERSS: 
        print("Customers:", NEW_CUSTOMERS)
        for _ in range(500):
            waiting_time = []
            RANDOM_SEED = random.randint(1, 100000000)

            def source(env, number, interval, counter):
                """Source generates customers randomly"""
                for i in range(number):
                    c = customer(env, 'Customer%02d' % i, counter, i)
                    env.process(c)
                    t = random.expovariate(1/interval)
                    yield env.timeout(t)

            def customer(env, name, counter, i):
                """Customer arrives, is served and leaves."""
                job_time = np.random.uniform(0,1)

                if job_time < 0.25:
                    jb = random.expovariate(1/22.5)
                else:
                    jb = random.expovariate(1/4.5)

                arrive = env.now

                with counter.request() as req:
                    # Wait for the counter
                    yield req

                    wait = env.now - arrive
                    waiting_time.append(wait)
                    yield env.timeout(jb)

            # Setup and start the simulation
            random.seed(RANDOM_SEED)
            env = simpy.Environment()
        
            # Start processes and run
            counter = simpy.Resource(env, capacity=C)
            env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/C, counter))
            env.run()

            c_values.append(np.mean(waiting_time))
            # print("wait", np.mean(waiting_time))
            c_group.append(f"{C} server(s)")
            Customer.append(NEW_CUSTOMERS)

data = {'Servers':c_group, "Values":c_values, "Amount of Customers":Customer}
df = pd.DataFrame(data) 
df
df.to_csv("Longtail_values10.csv")