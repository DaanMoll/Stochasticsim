import numpy as np
import matplotlib.pyplot as plt 

def making_connections(cities, connections, total_cities, city_count):
    """
    Makes initial random connections between cities
    Checks the distances between cities and save them
    """
    total_distance = []
    counter = 0
    while len(connections) != city_count -1 and len(total_cities)!= 102:
        city1 = np.random.randint(1,city_count)
        city2 = np.random.randint(1,city_count)
        counter += 1
        if counter >= 10000:
            return 0
        while city1 == city2:
            city2 = np.random.randint(1,city_count)

        city1 = cities[city1]
        city2 = cities[city2]

        if (city1, city2) in connections or total_cities.count(city1) == 2 or total_cities.count(city2) ==2:
            continue


        distance = np.sqrt((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)
        connections.append((city1, city2, distance))
        total_distance.append(distance)
        total_cities.append(city1)
        total_cities.append(city2)
        plt.plot([city1[0],city2[0]], [city1[1],city2[1]])

    # print(len(total_cities))
    return np.sum(total_distance), connections, total_cities