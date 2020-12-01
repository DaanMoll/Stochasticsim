from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import simpy

amount_of_customers = 100000
interval_customers = 10
server_values = []
server_group = []
customers = []
amount_of_servers = [1, 2, 4]

for servers in amount_of_servers:
    all_waits = []
    for _ in range(500):
        waiting_time = []
        RANDOM_SEED = random.randint(1, 100000000)

        def source(env, number, interval, counter):
            """Source generates customers randomly"""
            for i in range(number):
                c = customer(env, 'Customer%02d' % i, counter, i, service_time=9.0)
                env.process(c)
                t = random.expovariate(1/interval)
                yield env.timeout(t)

        def customer(env, name, counter, i, service_time):
            """Customer arrives, is served and leaves."""
            arrive = env.now

            with counter.request() as req:
                yield req

                wait = env.now - arrive
                waiting_time.append(wait)

                st = service_time
                yield env.timeout(st)
                

        # setup and start the simulation
        random.seed(RANDOM_SEED)
        env = simpy.Environment()
        counter = simpy.Resource(env, capacity=servers)
        env.process(source(env, amount_of_customers, interval_customers/servers, counter))
        env.run()

        server_values.append(np.mean(waiting_time))
        server_group.append(f"{servers} server(s)")
        customers.append(amount_of_customers)

data = {'Servers':server_group, "Values":server_values, "Amount of Customers":customers}
df = pd.DataFrame(data) 
df
df.to_csv("Waiting_data/MDC_values10.csv")
