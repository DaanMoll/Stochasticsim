import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random
import time

def acceptance_probability(distance, new_distance, temperature):
    """
    Calculate probability of accepting new cost
    """
    if new_distance < distance:
        return 1
    else:
        p = np.exp(- (new_distance - distance) / temperature)
        return p


    # print(current_temp)
    return current_temp
def hill_climber(distance, connections):
    distancess = []
    fout = 0
    iteration = 0
    current_distance = distance

    replace_cities = []
    replace_connections = 2
    max_iterations = 100

    while (max_iterations -1) != iteration:
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
        # print(iteration)
        if len(check_connections) > 1:

            # print("FOUTE SOL \n", check_connections, len(check_connections))
            connection = backup_connections
            fout += 1
            connections = backup_connections
            continue
        iteration += 1 
        if new_distance < current_distance:
            current_distance = new_distance
        else:        
            connections = backup_connections
        distancess.append(current_distance)
    print("current:", current_distance)
    print("fout van:", fout, "iterations:", max_iterations)

    return connections, current_distance, distancess

def simulated_annealing(distance, connections):
    distancess = []
    fout = 0
    iteration = 0
    start_temp = 10
    current_distance = distance

    replace_cities = []
    replace_connections = 2
    max_iterations = 100

    while (max_iterations -1) != iteration:
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
        # print(iteration)
        if len(check_connections) > 1:

            # print("FOUTE SOL \n", check_connections, len(check_connections))
            connection = backup_connections
            fout += 1
            connections = backup_connections
            continue
        iteration += 1 
        if new_distance < current_distance:
            current_distance = new_distance
        else:        
            current_temp = start_temp - (start_temp/max_iterations) * iteration
            ap = acceptance_probability(current_distance, new_distance, current_temp)
            if random.uniform(0, 1) < ap:
                current_distance = new_distance
            else:
                connections = backup_connections
        distancess.append(current_distance)
    print("current:", current_distance)
    print("fout van:", fout, "iterations:", max_iterations)

    return connections, current_distance, distancess

def simulated_annealing_2opt(distance, cities, order):
<<<<<<< HEAD
    kind  = "Linear"
    start_temp = 500
    max_iterations = 100
=======
    start_temp = 1
    max_iterations = 1
>>>>>>> 7c86b702449e8c9956bbf81c9af911b6ca972862
    accepted = 0
    count = 0
    iteration = 0
    same = 0

    distances = []
    current_distance = distance
    # print(cities)
<<<<<<< HEAD
    while current_distance > 450 and iteration < 500:    
        print("Iteration: ", iteration)
        print("Current distance: ",current_distance)
        print("Same: ", same)
        iteration += 1
        current_distance1 = current_distance
        
=======
    start_time = time.time()

    for iteration in range(max_iterations):
        print("iteration:", iteration)
        print("time for iter:", time.time() - start_time)
        start_time = time.time()
>>>>>>> 7c86b702449e8c9956bbf81c9af911b6ca972862
        for i in range(0, len(order) - 2):
            for j in range(i + 1, len(order)):
             
                backup_order = copy.deepcopy(order)
                connections = []
                # print(len(order))
                if j - i == 1: continue

                switch_1 = order[i]
                index_1 = i
                switch_2 = order[j]
                index_2 = j
                
                order[index_1] = switch_2
                order[index_2] = switch_1

                new_order = copy.deepcopy(order)
                # print(order)
                first_city = order[0]
                current_city = order[0]
                order.pop(0)
                
                # print(len(order))
                while len(order) != 0:
                    # print(order)
                    city1 = cities[current_city]
                    city2 = cities[order[0]]
                    distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
                    connections.append((current_city, order[0], distance))
                    current_city = order[0]
                    order.pop(0)


                # hier nog weer lijn naar start city
                city1 = cities[current_city]
                city2 = cities[first_city]
                distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
                connections.append((current_city, first_city, distance))

                new_distance = 0
                for connection in connections:
                    new_distance += float(connection[2])
                # print(new_distance)
<<<<<<< HEAD
                current_temp = cooling_schedule(start_temp, max_iterations, iteration, kind)
=======

                current_temp = start_temp - (start_temp/max_iterations) * iteration
>>>>>>> 7c86b702449e8c9956bbf81c9af911b6ca972862
                ap = acceptance_probability(current_distance, new_distance, current_temp)
                if random.uniform(0, 1) < ap:
                    current_distance = new_distance
                    order = new_order
                    accepted += 1
                else:
                    order = backup_order
                distances.append(current_distance)
                count+=1
        print("Current temperature: ",current_temp)
    
    if current_distance == current_distance1:
        same+=1
    else:
        same = 0
        
    print("accepted:", accepted)
    print(current_distance)
    return connections, current_distance, distances, count

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
    total_distance.append(distance)

    return connections, np.sum(total_distance), order

def opt_order(cities):
    order = []
    connections = []

    with open("TSP_data/a280.opt.tour.txt") as data:
        for row in data:
            row = row.strip("\n")
            if row.isdigit():
                order.append(int(row))
    order = [38, 33, 32, 44, 26, 35, 7, 3, 12, 17, 27, 37, 24, 21, 10, 0, 5, 15, 13, 25, 20, 36, 8, 31, 18, 1, 45, 11, 39, 14, 40, 29, 19, 50, 6, 4, 9, 47, 43, 28, 46, 34, 16, 49, 48, 30, 42, 23, 41, 22, 2]
    first_city = order[0]
    print("first", first_city)
    current_city = order[0]
    order.pop(0)

    while len(order) != 0:
        city1 = cities[current_city]
        city2 = cities[order[0]]
        distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
        connections.append((current_city, order[0], distance))
        current_city = order[0]
        order.pop(0)

    # hier nog weer lijn naar start city
    city1 = cities[current_city]
    city2 = cities[first_city]
    distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
    connections.append((current_city, first_city, distance))

    new_distance = 0
    for connection in connections:
        new_distance += float(connection[2])

    print("opt dist:", new_distance)

    # for connection in connections:
    #     plt.plot([cities[connection[0]][0], cities[connection[1]][0]], [cities[connection[0]][1], cities[connection[1]][1]])
    # plt.show()


if __name__ == "__main__":
    # Choose TSP file
    file_ = "eil51.tsp.txt"

    # Initializes lists and dictionaries
    cities = {}

    # Reads from data file and plots and saves all coordinates from cities
    with open(f'TSP_data/{file_}', 'r') as reader:
        counter = 0
        for line in reader:
            line = line.strip("\n")
            line = line.split()
    
            if line[0].isdigit():
                # counter+=1
                plt.plot(int(line[1]), int(line[2]), '.')
                cities[counter] = (int(line[1]), int(line[2]))
                counter+=1
                

    opt_order(cities)
    exit()


    result = nearest_neighbour(cities)
    original_connections = result[0]
    print(len(cities))
    start_distance = result[1]
    order = result[2]
   
    print("Sum before sa:", start_distance)
    
    result = simulated_annealing_2opt(np.sum(start_distance), cities, order)
    connections1 = result[0]
    sa_distance1 = result[1]
    distances1 = result[2]
    count = result[3]
    
    if sa_distance1 < start_distance:
        for connection in connections1:
            plt.plot([cities[connection[0]][0], cities[connection[1]][0]], [cities[connection[0]][1], cities[connection[1]][1]])
        plt.show()
        x = np.linspace(0, count-1, count)
        plt.plot(x, distances1)
        plt.show()
    
    
    # if sa_distance < start_distance:
    #     for connection in original_connections:
    #         plt.plot([cities[connection[0]][0], cities[connection[1]][0]], [cities[connection[0]][1], cities[connection[1]][1]])
    #     plt.show()
    #     x = np.linspace(0, 100, 99)
    #     plt.plot(x, distances)
    #     plt.show()