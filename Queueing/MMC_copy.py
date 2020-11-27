import scikit_posthocs as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import numpy as np
import simpy

waiting_time = []

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
    jb = random.expovariate(1/job_time)

    with counter.request() as req:
        yield req
        wait = env.now - arrive

        waiting_time.append(wait)

        yield env.timeout(jb)

NEW_CUSTOMERS = 1000
INTERVAL_CUSTOMERS = 10 

c_values = []
c_group = []
cs = [1,2,4]

for C in cs:
    for _ in range(500):
        waiting_time = []
        RANDOM_SEED = random.randint(1, 6000)

        random.seed(RANDOM_SEED)
        env = simpy.Environment()

        counter = simpy.resources.resource.Resource(env, capacity=C)
        env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/C, counter))
        env.run()

        c_values.append(np.mean(waiting_time))
        c_group.append(f"{C} server(s)")
    
data = {'Servers':c_group, "Values":c_values}
df = pd.DataFrame(data) 
df
df.to_csv("MMC_values.csv")

print("Done")