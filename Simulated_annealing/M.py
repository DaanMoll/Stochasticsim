import matplotlib.pyplot as plt 
import numpy as np

# Choose TSP file
file_ = "eil51.tsp.txt"

# Initializes lists and dictionaries
cities = {}
x_coordinates = []
y_coordinates = []
connections = []
total_distance = []
total_cities = []

# Reads from data file and plots and saves all coordinates from cities
with open(f'TSP_data\{file_}', 'r') as reader:
    counter = 1
    for line in reader:
        if line[0].isdigit():
            new_line = line.split()
            plt.plot(int(new_line[1]), int(new_line[2]), '.')
            cities[counter] = (int(new_line[1]), int(new_line[2]))
            counter+=1

# Makes initial random connections between cities
# Checks the distances between cities and save them
while len(connections) != counter - 1:
    city1 = np.random.randint(1,counter)
    city2 = np.random.randint(1,counter)
    while city1 == city2:
        city2 = np.random.randint(1,counter)

    city1 = cities[city1]
    city2 = cities[city2]

    if (city1, city2) in connections or total_cities.count(city1) == 2 or total_cities.count(city2) ==2:
        continue
    distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
    print(city1, city2)
    print(len(connections))
    connections.append((city1, city2))
    total_distance.append(distance)
    total_cities.append(city1)
    total_cities.append(city2)
    plt.plot([city1[0],city2[0]], [city1[1],city2[1]])

# Shows the connections and total distance
print(np.sum(total_distance))
plt.show()

