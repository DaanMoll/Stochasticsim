import random
import simpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NEW_CUSTOMERSS = [100000]
INTERVAL_CUSTOMERS = 10

c_values = []
c_group = []
Customers = []
cs = [1, 2, 4]

for C in cs:
    for NEW_CUSTOMERS in NEW_CUSTOMERSS: 
        print(NEW_CUSTOMERS)   
        all_waits = []
        for _ in range(500):
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
                # print('%7.4f %s: Here I am' % (arrive, name))

                with counter.request() as req:
                    # Wait for the counter
                    yield req

                    wait = env.now - arrive

                    waiting_time.append(wait)

                    # We got to the counter
                    # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
                    jb = job_time
                    yield env.timeout(jb)
                    
                    # print('%7.4f %s: Finished,' % (env.now, name), 'Helped time: %7.4f' % jb)    

            # Setup and start the simulation
            random.seed(RANDOM_SEED)
            env = simpy.Environment()

            # Start processes and run
            counter = simpy.Resource(env, capacity=C)
            env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/C, counter))
            env.run()

            # all_waits.append(round(np.mean(waiting_time), 2))

            c_values.append(np.mean(waiting_time))
            c_group.append(f"{C} server(s)")
            Customers.append(NEW_CUSTOMERS)

        # plt.hist(all_waits, label=C)

data = {'Servers':c_group, "Values":c_values, "Amount of Customers":Customers}
df = pd.DataFrame(data) 
df
df.to_csv("MDC_values10.csv")

# plt.legend()
# plt.show()