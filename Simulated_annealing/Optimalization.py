import numpy as np
import matplotlib.pyplot as plt 
from operator import itemgetter
from Making_connections import making_connections

def optimalization(total_distance, connections, total_cities, city_count, iterations, new_lines, cities):
    """
    Optimizes the total distance of the TSP with a random hillclimber
    returns the new connections and a list of all the distances

    """

    # Initializes distances
    distances = []

    # Iterates till all iterations are done
    for i in range(iterations):
        connections1 = connections
        total_cities1 = total_cities
        total_distance1 = total_distance
        deleted_connections = 0
        distances.append(total_distance)

        # Deletes lines
        
        connections = sorted(connections,key=itemgetter(2), reverse=True) 
        # city_count("\n\n Connections:\n", connections, "\n\n")
        # city_count("Cities:\n",len(total_cities), "\n")

        values=0
        while values == 0 or len(total_cities) == 100:
            # city_count("hallo")
            # Deletes lines and makes new lines
            # city_count("Cities:\n",len(total_cities), "\n", len(connections), "\n")
            for i in range(new_lines):
                deleted_connection = connections[i]
                total_cities.remove(deleted_connection[0])
                total_cities.remove(deleted_connection[1])
                connections.remove(deleted_connection)

            # Checks the new total distance
            # print(values)
            values = making_connections(cities, connections, total_cities, city_count)
            
            # city_count("A", len(total_cities))

    

        # Checks the new total distance
        values = making_connections(cities, connections, total_cities, city_count)
        total_distance = 0
        for con in values[1]:
            total_distance+=con[2]

        # If new distnace is smaller than old distance the new route is chosen
        if total_distance<total_distance1:    
            total_distance = total_distance
            connections = values[1]
            total_cities = values[2]
        else:
            connections = connections1
            total_distance = total_distance1
            total_cities = total_cities1

    return connections,distances

def optimalization1(total_distance, connections, total_cities, city_count, iterations, new_lines, cities):
    """
    Optimizes the total distance of the TSP with a random hillclimber
    returns the new connections and a list of all the distances

    """

    # Initializes distances
    distances = []

    # Iterates till all iterations are done
    for i in range(iterations):

        connections1 = connections
        total_cities_A = total_cities
        total_distance1 = total_distance
        distances.append(total_distance)
        values = 0

        while values == 0 or len(total_cities) == 100:
            # Deletes lines and makes new lines
            # city_count("Cities:\n",len(total_cities), "\n", len(connections), "\n")
            for r in range(new_lines):            
                choice = np.random.randint(0, len(connections))
                deleted_connection = connections1[choice]
                connections.remove(deleted_connection)
                total_cities.remove(deleted_connection[0])
                total_cities.remove(deleted_connection[1])

            # Checks the new total distance
            values = making_connections(cities, connections, total_cities, city_count)
            # city_count("A", len(total_cities))
        
        # city_count("B", len(total_cities))
        total_distance = 0
        for con in values[1]:
            total_distance+=con[2]

        # If new distnace is smaller than old distance the new route is chosen
        if total_distance<total_distance1:    
            total_distance = total_distance
            connections = values[1]
            total_cities = values[2]
        else:
            connections = connections1
            total_distance = total_distance1
            total_cities = total_cities_A
            
        
    return connections,distances
