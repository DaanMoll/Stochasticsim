import random
import simpy
import numpy as np
from matplotlib import pyplot as plt

c = 1
runtime = 1000
jobs = [0] * 50
waiting = [0] * 200
RANDOM_SEED = random.randint(1, 600)
# Total number of customers
NEW_CUSTOMERS = 500  
# Generate new customers roughly every x seconds
INTERVAL_CUSTOMERS = 2.1

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
        jb = random.expovariate(1/5)
    else:
        jb = random.expovariate(1)

    jobs[round(jb)] += 1
    arrive = env.now
    print('%7.4f %s: Here I am' % (arrive, name))

    with counter.request() as req:
        # Wait for the counter
        yield req

        wait = env.now - arrive

        # We got to the counter
        print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
        yield env.timeout(jb)
        
        print('%7.4f %s: Finished,' % (env.now, name), 'Helped time: %7.4f' % jb)


# Setup and start the simulation
random.seed(RANDOM_SEED)
env = simpy.Environment()

# Start processes and run
counter = simpy.Resource(env, capacity=c)
env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/c, counter))
env.run()

plt.bar(np.linspace(0, 20, 20), jobs[:20])
plt.show()
