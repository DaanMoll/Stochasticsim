from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import scikit_posthocs as sp
import scipy.stats as stats
import seaborn as sns
import simpy

amount_of_customers = 100000
interval_customers = 10
server_values = []
server_group = []
customers = []
amount_of_servers = [1,2,4]

for servers in amount_of_servers: 
    for _ in range(500):
        waiting_time = []
        RANDOM_SEED = random.randint(1,100000000)
        
        def source(env, number, interval, counter):
            """Source generates customers randomly"""
            for i in range(number):
                c = customer(env, 'Customer%02d' % i, counter, service_time=9.0)
                env.process(c)
                t = random.expovariate(1/interval)
                yield env.timeout(t)

        def customer(env, name, counter, service_time):
            """Customer arrives, is served and leaves."""
            arrive = env.now
            st = random.expovariate(1/service_time)
            
            with counter.request() as req:
                # Wait for the counter
                yield req
                wait = env.now - arrive
                waiting_time.append(wait)

                yield env.timeout(st)

        # setup and start the simulation
        random.seed(RANDOM_SEED)
        env = simpy.Environment()
        counter = simpy.resources.resource.Resource(env, capacity=servers)
        env.process(source(env, amount_of_customers, interval_customers/servers, counter))
        env.run()

        customers.append(f"{amount_of_customers}")
        server_values.append(np.mean(waiting_time))
        server_group.append(f"{servers} server(s)")

data = {'Servers':server_group, "Values":server_values, "Amount of Customers":customers}
df = pd.DataFrame(data) 
df
df.to_csv("Waiting_data/MMC_values10.csv")