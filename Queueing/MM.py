from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import simpy

servers = 1
interval_customers = 10
amount_of_customers_list = [25000, 50000, 75000, 100000]
rho_values = []
rho_group = []
customers = []
service_times = [1, 3, 6, 9]

for amount_of_customers in amount_of_customers_list:
    for service_time in service_times:
        for _ in range(500):
            waiting = []
            RANDOM_SEED = random.randint(1, 100000000)

            def source(env, number, interval, counter):
                """Source generates customers randomly"""
                for i in range(number):
                    c = customer(env, 'Customer%02d' % i, counter, i, service_time=service_time)
                    env.process(c)
                    t = random.expovariate(1/interval)
                    yield env.timeout(t)

            def customer(env, name, counter, i, service_time):
                """Customer arrives, is served and leaves."""
                arrive = env.now

                with counter.request() as req:
                    # wait for the counter
                    yield req

                    wait = env.now - arrive
                    waiting.append(wait)

                    st = random.expovariate(1/service_time)
                    yield env.timeout(st)
                    
            # setup and start the simulation
            random.seed(RANDOM_SEED)
            env = simpy.Environment()
            counter = simpy.Resource(env, capacity=servers)
            env.process(source(env, amount_of_customers, interval_customers/servers, counter))
            env.run()

            # append data for pd frames
            rho_values.append(np.mean(waiting))
            rho_group.append(f" Value: {service_time/interval_customers}")
            customers.append(f"{amount_of_customers}")

    data = {'Rho':rho_group, "Values":rho_values, "Amount of Customers":customers}
    df = pd.DataFrame(data) 
    df
    df.to_csv("Waiting_data/MM_values25k-100k10000.csv")