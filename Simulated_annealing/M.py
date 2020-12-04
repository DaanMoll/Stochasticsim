import matplotlib.pyplot as plt 
import numpy as np
from Making_connections import making_connections
from Optimalization import optimalization, optimalization1

# Choose TSP file
file_ = "eil51.tsp.txt"

# Initializes lists and dictionaries
connections = []
total_cities = []


def plot_cities():
    """
    Reads from data file and plots and saves
    all coordinates from cities. Returns amount of cities
    and the cities with coordinates.
    """

    # Initialize cities
    cities = {}

    # Opens file, reads lines, saves cities and makes a plot of the cities
    with open(f'TSP_data\{file_}', 'r') as reader:
        city_count = 1
        for line in reader:
            if line[0].isdigit():
                new_line = line.split()
                plt.plot(int(new_line[1]), int(new_line[2]), '.')
                cities[city_count] = (int(new_line[1]), int(new_line[2]))
                city_count+=1
    return city_count, cities

# Make values
city_count = plot_cities()[0]
cities = plot_cities()[1]

# Makes connections between the initial points
values = making_connections(cities, connections, total_cities, city_count)

# Returnes values of the connection fuction
total_distance = values[0]
connections = values[1]
total_cities = values[2]

# Shows the connections and total distance
plt.show()
print(total_distance)

# Choose the iterations of simulation
iterations = 100
new_lines = 10

# Optimizes the route of TSP
connections = optimalization(total_distance, connections, total_cities,city_count, iterations, new_lines, cities)

plt.close()

# Makes a plot of the new, shortest, route
plot_cities()
for con in connections[0]:
    plt.plot([con[0][0], con[1][0]],[con[0][1], con[1][1]])
plt.show()

x = np.linspace(0, (iterations-1), iterations)

plt.plot(x, connections[1])
plt.show()