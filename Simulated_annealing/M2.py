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
with open(f'TSP_data/{file_}', 'r') as reader:
    counter = 0
    for line in reader:
        if line[0].isdigit():
            counter+=1
            new_line = line.split()
            # print(new_line)
            plt.plot(int(new_line[1]), int(new_line[2]), '.')
            cities[counter] = (int(new_line[1]), int(new_line[2]))
            

# Makes initial random connections between cities
# Checks the distances between cities and save them
city_numbers = [i for i in range(1, 51, 1)]

while len(connections) != counter-1:
    np.random.shuffle(city_numbers)
    number_1 = city_numbers[0]

    city1 = cities[number_1]
    lowest_distance = 999999

    for number in city_numbers:
        if number_1 == number:
            continue

        city2 = cities[number]

        if (city1, city2) in connections or total_cities.count(city1) == 2 or total_cities.count(city2) == 2:
            continue

        distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
        
        if distance < lowest_distance:
            lowest_distance = distance
            number_2 = number
    
    # print(city1, city2)
    if lowest_distance == 999999:
        continue

    connections.append((number_1, number_2))

    total_distance.append(lowest_distance)

    total_cities.append(number_1)
    total_cities.append(number_2)

    if total_cities.count(number_1) == 2:
        city_numbers.remove(number_1)

    if total_cities.count(number_2) == 2:
        city_numbers.remove(number_2)

    plt.plot([city1[0],city2[0]], [city1[1],city2[1]])

# Shows the connections and total distance
count = [0] * counter
for connection in connections:
    count[int(connection[0])] += 1
    count[int(connection[1])] += 1

print(count[1:counter])
print(connections)

print(np.sum(total_distance))
plt.show()

