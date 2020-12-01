from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import simpy

total_customers = 10000
interval_customers = 10
server_values = []
server_group = []
customers = []
amount_of_servers = [1, 2, 4]

for servers in amount_of_servers:
    waiting_time = []
    for _ in range(500):
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
            # RNG to decide which exponential distribution to follow
            service_time = np.random.uniform(0,1)
            if service_time < 0.25:
                st = random.expovariate(1/22.5)
            else:
                st = random.expovariate(1/4.5)

            arrive = env.now

            with counter.request() as req:
                # wait for the server
                yield req

                wait = env.now - arrive
                waiting_time.append(wait)
                yield env.timeout(st)

        # setup and start the simulation
        random.seed(RANDOM_SEED)
        env = simpy.Environment()
    
        # start processes and run
        counter = simpy.Resource(env, capacity=servers)
        env.process(source(env, total_customers, interval_customers/servers, counter))

        env.run()

        # append data for pd frame
        server_values.append(np.mean(waiting_time))
        server_group.append(f"{servers} server(s)")
        customers.append(total_customers)

data = {'Servers':server_group, "Values":server_values, "Amount of Customers":customers}
df = pd.DataFrame(data) 
df
df.to_csv("Waiting_data/MHC_values.csv")