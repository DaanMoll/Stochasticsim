import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random

def acceptance_probability(distance, new_distance, temperature):
    """
    Calculate probability of accepting new cost
    """
    if new_distance < distance:
        return 1
    else:
        p = np.exp(- (new_distance - distance) / temperature)
        return p

def simulated_annealing(distance, connections):
    fout = 0

    start_temp = 100
    current_distance = distance

    replace_cities = []
    replace_connections = 2
    max_iterations = 1000

    for iteration in range(max_iterations):
        backup_connections = copy.deepcopy(connections)

        distance_order = sorted(connections, key=itemgetter(2), reverse=True)

        for _ in range(replace_connections):
            replace_cities.append(distance_order[0][0])
            replace_cities.append(distance_order[0][1])

            connections.remove(distance_order[0])
            distance_order.pop(0)

            random_city = random.choice(distance_order)
            replace_cities.append(random_city[0])
            replace_cities.append(random_city[1])
            connections.remove(random_city)
            distance_order.remove(random_city)

        while len(replace_cities) != 0:
            number_1 = np.random.choice(replace_cities)

            for city in replace_cities:
                if replace_cities.count(city) == 2:
                    number_1 = city
                    break

            lowest_distance = 999999
            replace_cities.remove(number_1)
            for number in replace_cities:
                if number == number_1:
                    continue

                city1 = cities[number_1]
                city2 = cities[number]

                distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)

                if distance < lowest_distance:
                    lowest_distance = distance
                    number_2 = number
            
            if lowest_distance == 999999:
                continue

            replace_cities.remove(number_2)
            connections.append((number_1, number_2, distance))

        new_distance = 0
        for connection in connections:
            new_distance += float(connection[2])

        # check if new solution is correct
        check_connections = copy.deepcopy(connections)
        position = check_connections[0]

        while check_connections.count(position) > 0:
            next_position = position[1]
            check_connections.remove(position)

            for connection in check_connections:
                if next_position == connection[0]:
                    position = connection
        
        if len(check_connections) > 1:
            fout += 1
            connections = backup_connections
            continue
        
        if new_distance < current_distance:
            current_distance = new_distance
        else:
            current_temp = start_temp - (start_temp/max_iterations) * iteration
            ap = acceptance_probability(current_distance, new_distance, current_temp)
            if random.uniform(0, 1) < ap:
                current_distance = new_distance
            else:
                connections = backup_connections

    print("current:", current_distance)
    print("fout van:", fout, "iterations:", max_iterations)

    return connections, current_distance

def nearest_neighbour(cities):
    order = []
    connections = []
    total_distance = []

    city_numbers = [i for i in range(1, counter+1, 1)]

    number_1 = np.random.choice(city_numbers)
    first_city = number_1

    order.append(first_city)

    city_numbers.remove(number_1)

    while len(city_numbers) != 0:
        city1 = cities[number_1]
        
        lowest_distance = 999999

        for number in city_numbers:
            city2 = cities[number]
            distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
            
            if distance < lowest_distance:
                lowest_distance = distance
                number_2 = number
                
        if lowest_distance == 999999:
            continue

        connections.append((number_1, number_2, lowest_distance))
        order.append(number_2)
        city_numbers.remove(number_2)
        total_distance.append(lowest_distance)
        number_1 = number_2

    city1 = cities[number_1]
    city2 = cities[first_city]

    distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)

    connections.append((number_1, first_city, distance))
    order.append(first_city)
    total_distance.append(distance)

    return connections, np.sum(total_distance)

if __name__ == "__main__":
    # Choose TSP file
    file_ = "eil51.tsp.txt"

    # Initializes lists and dictionaries
    cities = {}

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
            

    result = nearest_neighbour(cities)
    connections = result[0]
    start_distance = result[1]
    print("Sum before sa:", start_distance)

    result = simulated_annealing(np.sum(start_distance), connections)
    connections = result[0]
    sa_distance = result[1]
    
    if sa_distance < start_distance:
        for connection in connections:
            plt.plot([cities[connection[0]][0], cities[connection[1]][0]], [cities[connection[0]][1], cities[connection[1]][1]])
        plt.show()

    