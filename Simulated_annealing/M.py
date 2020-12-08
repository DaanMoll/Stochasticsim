import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random
import math
import time

def make_matrix(tsp_file):
    """"
    Creates an adjacency matrix based on the tsp file
    """ 
    # Extracting node coordinates from tsp file
    cities = {}
    counter = 0
    node_list = []

    with open(f"TSP_data/{tsp_file}.tsp.txt","r") as reader:
        for line in reader:
            line = line.strip("\n")
            line = line.split()

            if line[0].isdigit():

                cities[counter] = (line[1], line[2])
                counter +=1 
                node_list.append([int(x) for x in line])

    # Creating adjacency matrix
    num_node = len(node_list)
    adjacency_matrix = np.zeros((num_node,num_node))
    for node1 in range(num_node):
        for node2 in range(num_node):
            if node1 != node2:
                x = abs(node_list[node1][1] - node_list[node2][1])
                y = abs(node_list[node1][2] - node_list[node2][2])
                dist = np.sqrt(x**2+y**2)
                adjacency_matrix[node1][node2] = dist

    return adjacency_matrix, cities

# def acceptance_probability(distance, new_distance, temperature):
#     """
#     Calculate probability of accepting new cost
#     """
#     if new_distance < distance:
#         return 1
#     else:
#         p = np.exp(- (new_distance - distance) / temperature)
#         return p

def nearest_neighbour(matrix):
    order = []
    total_distance = 0
    print("matrix len", len(matrix))
    city_numbers = [i for i in range(0, len(matrix), 1)]

    number_1 = np.random.choice(city_numbers)
    first_city = number_1

    order.append(first_city)

    city_numbers.remove(number_1)
    number_2 = number_1

    while len(city_numbers) != 0:
        closest = math.inf

        for number in city_numbers:
            if number == number_1:
                continue
            distance = matrix[number_1][number]

            if distance < closest:
                closest = distance
                number_2 = number
        
        if number_2 == number_1:
            continue

        distance = matrix[number_1][number_2]
        total_distance += distance
    
        order.append(number_2)
        city_numbers.remove(number_2)
        number_1 = number_2

    # end to begin
    distance = matrix[number_2][first_city]
    total_distance += distance

    return total_distance, order

def simulated_annealing_2opt(matrix, order, distance):
    start_temp = 10
    max_iterations = 10
    accepted = 0
    count = 0

    distances = []
    current_distance = distance

    start_time = time.time()

    for iteration in range(max_iterations):
        print("iteration:", iteration)
        print("time for iter:", time.time() - start_time)
        start_time = time.time()
        for i in range(0, len(order) - 2):
            for j in range(i + 1, len(order)):
                backup_order = copy.deepcopy(order)
                distance = 0
                if j - i == 1: continue

                switch_1 = order[i]
                index_1 = i
                switch_2 = order[j]
                index_2 = j
                
                order[index_1] = switch_2
                order[index_2] = switch_1

                new_order = copy.deepcopy(order)
                
                # calculate cost new order
                first_city = order[0]
                current_city = order[0]
                order.pop(0)
                while len(order) > 0:
                    next_city = order[0]
                    order.pop(0)

                    distance += matrix[current_city][next_city]
                    current_city = next_city
                #line from end to start
                distance += matrix[current_city][first_city]

                current_temp = start_temp - (start_temp/max_iterations) * iteration
                ap = acceptance_probability(current_distance, distance, current_temp)
                
                if random.uniform(0, 1) < ap:
                    # print("accepted:", current_city, "new:", distance)
                    current_distance = distance
                    order = new_order
                    accepted += 1
                else:
                    order = backup_order
                distances.append(current_distance)
                count+=1
        
    print("accepted:", accepted)
    return current_distance, distances, count

def opt_order(tsp_file, matrix):
    distance = 0
    order = []

    with open(f"TSP_data/{tsp_file}.opt.tour.txt") as data:
        for row in data:
            row = row.strip("\n")
            if row.isdigit():
                order.append(int(row))

    first_city = order[0] - 1
    current_city = order[0] - 1
    order.pop(0)

    while len(order) > 0:
        next_city = order[0] - 1
        order.pop(0)

        distance += matrix[current_city][next_city]
        current_city = next_city

    distance += matrix[current_city][first_city]
    
    return distance

if __name__ == "__main__":
    tsp_file = "a280"

    matrix = make_matrix(tsp_file)

    opt_distance = opt_order(tsp_file, matrix)
    print(f"{tsp_file}, optimal distance:", opt_distance)

    result = nearest_neighbour(matrix)
    nn_distance = result[0]
    nn_order = result[1]
    print(f"{tsp_file}, nn distance:", nn_distance)

    start_time = time.time()
    print("start time:", start_time)

    result = simulated_annealing_2opt(matrix, nn_order, nn_distance)
    sa2opt_distance = result[0]
    print(sa2opt_distance)





