"""
Bank renege example

Covers:

- Resources: Resource
- Condition events

Scenario:
  A counter with a random service time and customers who renege. Based on the
  program bank08.py from TheBank tutorial of SimPy 2. (KGM)

"""

import random
import simpy


RANDOM_SEED = 42
c = 1
runtime = 50
# Total number of customers
# NEW_CUSTOMERS = 500
# Generate new customers roughly every x seconds
INTERVAL_CUSTOMERS = 10.0  
# Min. customer patience
MIN_PATIENCE = 1  
# Max. customer patience
MAX_PATIENCE = 3

def source(env, interval, counter):
    """Source generates customers randomly"""
    i = 0
    while True:
        c = customer(env, 'Customer%02d' % i, counter, time_in_bank=12.0)
        env.process(c)
        t = random.expovariate(1/interval) 
        yield env.timeout(t)
        
        i += 1


def customer(env, name, counter, time_in_bank):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    print('%7.4f %s: Here I am' % (arrive, name))

    with counter.request() as req:
        patience = random.uniform(MIN_PATIENCE, MAX_PATIENCE)
        # Wait for the counter or abort at the end of our tether
        results = yield req | env.timeout(patience)

        wait = env.now - arrive
        helped = env.now

        if req in results:
            # We got to the counter
            print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

            tib = random.expovariate(1.0 / time_in_bank)
            yield env.timeout(tib)
            print('%7.4f %s: Finished,' % (env.now, name), 'Helped time: %7.4f' % (env.now - helped))

        else:
            # We reneged
            print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, wait))


# Setup and start the simulation
print('Bank renege')
random.seed(RANDOM_SEED)
env = simpy.Environment()

# Start processes and run
counter = simpy.Resource(env, capacity=c)
env.process(source(env, INTERVAL_CUSTOMERS/c, counter))
env.run(runtime)