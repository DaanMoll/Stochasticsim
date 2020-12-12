import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random
import math
import pandas as pd
import sys

def plot_route(tsp_file, route):
    cities = {}

    # Reads from data file and plots and saves all coordinates from cities
    with open(f"TSP_data/{tsp_file}.tsp.txt","r") as reader:
        counter = 0
        for line in reader:
            line = line.strip("\n")
            line = line.split()
    
            if line[0].isdigit():
                plt.plot(int(line[1]), int(line[2]), '.')
                # plt.annotate(counter, [int(line[1]), int(line[2])], fontsize=8)

                cities[counter] = (int(line[1]), int(line[2]))
                counter+=1

    for i in range(len(route) - 1):
        city1 = cities[route[i]]
        city2 = cities[route[i+1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    city1 = cities[route[-1]]
    city2 = cities[route[0]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    # count = np.linspace(0, count - 1, count)
    # plt.plot(count, distances)
    plt.show()

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

def nearest_neighbour2(matrix):
    order = []
    total_distance = 0
    city_numbers = [i for i in range(0, len(matrix), 1)]

    number_1 = np.random.choice(city_numbers)
    first_city = number_1

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
    order.append(first_city)
    total_distance += distance
    return order

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

def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

def acceptance_probability(cost, temperature):
    """
    Calculate probability of accepting new cost
    """
    if cost < 0:
        return 1
    else:
        p = np.exp(- (cost) / temperature)
        return p

def cooling_schedule(start_temp, max_iterations, iteration, kind):
    if kind  == "Linear":
        # multiplicative
        alpha = 1
        current_temp = start_temp/(1 + alpha*iteration)
    elif kind == "Log":
        alpha = 50
        # multiplicative
        current_temp =  start_temp/(alpha * (math.log(iteration + 1, 10)))
    elif kind == "Exponential":
        # multiplicative
        current_temp = start_temp*0.9**iteration
    elif kind == "Quadratic":
        # multiplicative
        alpha = 1
        current_temp = start_temp/(1 + alpha * iteration**2)
    
    return current_temp

def sa_two_opt(route, cost_mat, distance, cooling, max_iterations, start_temp, markov_length):
    best = route
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        
        for _ in range(markov_length):
            i = random.randint(1, len(route) - 1)
            j = random.randint(1, len(route) - 1)

            while j == i or j-i == 1 or j-i == -1:
                j = random.randint(1, len(route) - 1)
                
            current_temp = cooling_schedule(start_temp, max_iterations, iteration, cooling)
            cost = cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j])
            ap = acceptance_probability(cost, current_temp)

            if random.uniform(0, 1) < ap:
                best[i:j] = best[j - 1:i - 1:-1]
                distance += cost

        route = best
    return best


if __name__ == '__main__':
    tsp_file = "a280"

    matrix = make_matrix(tsp_file)
    cost_mat = list(matrix)

    if len(sys.argv) <= 1:
        print("Usage: python markov.py coolingschedule")
        exit()

    cooling = sys.argv[1]

    temperatures = {}
    # temperatures["Exponential"] = [150, 220, 450]
    # temperatures["Linear"] = [530, 850, 1700]
    # temperatures["Log"] = [120, 180, 370]
    # temperatures["Quadratic"] = [530, 850, 1700]
    # percentages = [70, 80, 90]

    temperatures["Exponential"] = [220]
    temperatures["Linear"] = [850]
    temperatures["Log"] = [180]
    temperatures["Quadratic"] = [850]
    percentages = [80]

    iteration = 10000
    markov_length = [10, 25, 50, 75, 100, 125, 150]
    
    costs = []
    temperatures_v = []
    schedules = []
    percentage_v = []
    init_routes = []
    init_costs = []
    routes = []
    markovs = []
    
    print(cooling)
    temps = temperatures[cooling]

    for markov in markov_length:
        print(cooling, markov)
        for i in range(len(temps)):
            temp = temps[i]
            percentage = percentages[i]

            max_i = 301
            for i in range(1, max_i):
                nn_route = nearest_neighbour(matrix)
                distance_nn = calculate_cost(nn_route, matrix)
                init_routes.append(nn_route)
                init_costs.append(distance_nn)

                backup_nn = copy.deepcopy(nn_route)
                best_route = sa_two_opt(backup_nn, cost_mat, distance_nn, cooling, iteration, temp, markov)
                cost = calculate_cost(best_route, matrix)
                
                routes.append(best_route)
                costs.append(cost)
                percentage_v.append(percentage)
                schedules.append(cooling)
                markovs.append(markov)
    
    data = {"Cooling_schedule":schedules, "Markov":markovs, "Init cost":init_costs, "Cost":costs, "Percentage":percentage_v, "Routes":routes, "Init routes":init_routes}
    df = pd.DataFrame(data) 
    df
    df.to_csv(f"data/values_{cooling}_{max_i}_iter.csv")


