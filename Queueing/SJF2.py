import random
import simpy

c = 1
runtime = 100

waiting_time = []
RANDOM_SEED = random.randint(1, 600)
# Total number of customers
NEW_CUSTOMERS = 1100  
# Generate new customers roughly every x seconds
INTERVAL_CUSTOMERS = 10

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

    print('%7.4f %s: Here I am' % (arrive, name), '%7.4f My JB' % jb)

    with counter.request(priority=jb) as req:
        # Wait for the counter
        yield req

        wait = env.now - arrive
        helped = env.now

        # We got to the counter
        print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

        yield env.timeout(jb)
        
        print('%7.4f %s: Finished,' % (env.now, name), 'Helped time: %7.4f' % jb)


# Setup and start the simulation
random.seed(RANDOM_SEED)
env = simpy.Environment()

# Start processes and run
counter = simpy.resources.resource.PriorityResource(env, capacity=c)
env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS/c, counter))
env.run(runtime)