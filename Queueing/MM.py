import random
import simpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

c = 1
runtime = 100
# Total number of customers
NEW_CUSTOMERSS = [25000, 50000, 75000, 100000]
# Generate new customers roughly every x seconds
job_times = [1, 3, 6, 9]

wait_values = []
wait_group = []
customers = []

for NEW_CUSTOMERS in NEW_CUSTOMERSS:
    print(NEW_CUSTOMERS)
    for job_time in job_times:
        INTERVAL_CUSTOMERS = 10
        all_waits = []

        for _ in range(500):
            waiting = []
            RANDOM_SEED = random.randint(1, 100000000)

            def source(env, number, interval, counter):
                """Source generates customers randomly"""
                for i in range(number):
                    c = customer(env, 'Customer%02d' % i, counter, i, job_time=job_time)
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

                    waiting.append(wait)

                    # We got to the counter
                    # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
                    jb = random.expovariate(1/job_time)
                    yield env.timeout(jb)
                    
                    # print('%7.4f %s: Finished,' % (env.now, name), 'Helped time: %7.4f' % jb)    

            # Setup and start the simulation
            random.seed(RANDOM_SEED)
            env = simpy.Environment()

            # Start processes and run
            counter = simpy.Resource(env, capacity=c)
            env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/c, counter))
            env.run()

            # all_waits.append(np.mean(waiting))

            wait_values.append(np.mean(waiting))
            wait_group.append(f" Value: {job_time/INTERVAL_CUSTOMERS}")
            customers.append(f"{NEW_CUSTOMERS}")

        # plt.hist(all_waits, label=INTERVAL_CUSTOMERS)

    data = {'Rho':wait_group, "Values":wait_values, "Amount of Customers":customers}
    df = pd.DataFrame(data) 
    df
    df.to_csv("MM_values25k-100k.csv")

# plt.legend()
# plt.show()