import random
import numpy as np
import pandas as pd
import simpy
from matplotlib import pyplot as plt
import numpy as np

c = 1
NEW_CUSTOMERSS = [100000] 
c1 = []
simmulation = []
Customer = []
INTERVAL_CUSTOMERS = 10

for NEW_CUSTOMERS in NEW_CUSTOMERSS:
    for i in range(500):
        waiting_time = []
        RANDOM_SEED = random.randint(1, 100000000)

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
            jb = random.expovariate(1.0 / job_time)

            # print('%7.4f %s: Here I am' % (arrive, name), '%7.4f My JB' % jb)

            with counter.request(priority=jb) as req:
                # Wait for the counter
                yield req

                wait = env.now - arrive
                waiting_time.append(wait)
                helped = env.now
                # print(wait)
                # print(waiting_time)

                # We got to the counter
                # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

                yield env.timeout(jb)
                


        # Setup and start the simulation
        random.seed(RANDOM_SEED)
        env = simpy.Environment()

        # Start processes and run
        counter = simpy.resources.resource.PriorityResource(env, capacity=c)
        env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/c, counter))
        env.run()

        simmulation.append("SJF")
        c1.append(np.mean(waiting_time))
        Customer.append(NEW_CUSTOMERS)

data = {"Servers":simmulation,"Values":c1, "Amount of customers":Customer}
df = pd.DataFrame(data) 
df
df.to_csv("SJF_values2.csv")   
