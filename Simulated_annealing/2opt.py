import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random
import math

def cooling_schedule(start_temp, max_iterations, iteration, kind):
    alpha = 1.5
    if kind  == "Linear":
        current_temp = start_temp/(1 + alpha*iteration)
    elif kind == "Linear2":
        current_temp = start_temp/(start_temp/max_iterations) * iteration
    elif kind == "Log":
        current_temp =  start_temp/(1 + alpha * (math.log(start_temp + iteration, 10)))
    elif kind == "Exponential":
        current_temp = start_temp*0.9**iteration
    elif kind == "Quadratic":
        current_temp =  start_temp/(1 + alpha * iteration**2)
    return current_temp

def make_matrix(tsp_file):
    """"
    Creates an adjacency matrix based on the tsp file
    """ 
    # Extracting node coordinates from tsp file
    node_list = []
    with open(f"TSP_data/{tsp_file}.tsp.txt","r") as reader:
        for line in reader:
            line = line.strip("\n")
            line = line.split()

            if line[0].isdigit():
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

    return adjacency_matrix


def acceptance_probability(cost, temperature):
    """
    Calculate probability of accepting new cost
    """
    if cost < 0:
        return 1
    else:
        p = np.exp(- (cost) / temperature)
        return p

def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

def two_opt(route, cost_mat):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best

    return best

def sa_two_opt(route, cost_mat, distance):
    max_iterations = 1000
    kind = "Linear"
    start_temp = 150
    accepted = 0
    distances = []
    best = route
    count = 0

    for iteration in range(max_iterations):
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue

                current_temp = cooling_schedule(start_temp, max_iterations, iteration, kind)
                cost = cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j])
                ap = acceptance_probability(cost, current_temp)
                
                if random.uniform(0, 1) < ap:
                    best[i:j] = best[j - 1:i - 1:-1]
                    accepted += 1
                    distance += cost
                    
                distances.append(distance)
                count += 1

        route = best
    return best, distances, count

def calculate_cost(order, matrix):
    distance = 0
    order = copy.deepcopy(order)
    first_city = order[0]
    current_city = order[0]
    order.pop(0)
    
    while len(order) > 0:
        next_city = order[0]
        order.pop(0)

        distance += matrix[current_city][next_city]
        current_city = next_city

    distance += matrix[current_city][first_city]
    
    return distance

def nearest_neighbour(matrix):
    order = []
    total_distance = 0
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
    return order

if __name__ == '__main__':
    tsp_file = "a280"

    matrix = make_matrix(tsp_file)
    
    nodes = len(matrix)
    nn_route = nearest_neighbour(matrix)
    distance_nn = calculate_cost(nn_route, matrix)
    print("nn distance:", distance_nn)
    print("len nn", len(nn_route))

    cost_mat = list(matrix)
    result = sa_two_opt(nn_route, cost_mat, distance_nn)
    best_route = result[0]
    distances = result[1]
    count = result[2]
    
    print("na sa cost:", calculate_cost(best_route, matrix))
    
    # start plot
    # Initializes lists and dictionaries
    cities = {}

    # Reads from data file and plots and saves all coordinates from cities
    with open(f"TSP_data/{tsp_file}.tsp.txt","r") as reader:
        counter = 0
        for line in reader:
            line = line.strip("\n")
            line = line.split()
    
            if line[0].isdigit():
                # counter+=1
                plt.plot(int(line[1]), int(line[2]), '.')
                # plt.annotate(counter, [int(line[1]), int(line[2])], fontsize=8)

                cities[counter] = (int(line[1]), int(line[2]))
                counter+=1

    route = best_route
    for i in range(len(route) - 1):
        city1 = cities[route[i]]
        city2 = cities[route[i+1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    city1 = cities[route[-1]]
    city2 = cities[route[0]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    plt.show()
    count = np.linspace(0, count - 1, count)

    plt.plot(count, distances)
    plt.show()
