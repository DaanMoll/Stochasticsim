import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random
import math
import pandas as pd
import sys

def plot_route(tsp_file, route):
    """
    Plot TSP map with route given as input.
    """
    cities = {}

    # Reads from data file and plots and saves all coordinates from cities
    with open(f"TSP_data/{tsp_file}.tsp.txt","r") as reader:
        counter = 0
        for line in reader:
            line = line.strip("\n")
            line = line.split()
    
            if line[0].isdigit():
                plt.plot(int(line[1]), int(line[2]), '.')
                cities[counter] = (int(line[1]), int(line[2]))
                counter+=1

    for i in range(len(route) - 1):
        city1 = cities[route[i]]
        city2 = cities[route[i+1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    city1 = cities[route[-1]]
    city2 = cities[route[0]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]])
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
    """
    Nearest neighbour heuristic. Pick random starting city, followed by connecting to closes neighbour.
    """
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

def calculate_cost(order, matrix):
    """
    Calculates cost of given route/order
    """
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
    """
    Basic two opt algorithm
    """
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
    """
    Return change of cost after 2 opt has been applied
    """
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
    """
    Calculate current temperature with the according cooling schedule
    """
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
        alpha = 0.9
        current_temp = start_temp*alpha**iteration
    elif kind == "Quadratic":
        # multiplicative
        alpha = 1
        current_temp = start_temp/(1 + alpha * iteration**2)

    # print("doei", kind)
    return current_temp

def sa_two_opt(route, cost_mat, distance, cooling, max_iterations, start_temp, markov_length):
    """
    Simulated annealing algorithm, using the 2 opt algorithm to create new states.
    """
    best = route
    costs = []
    iter2 = []
    cooling2 = []
    count = 0
    counter = 0

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        # Remove 2 lines markov_lenght amount of tim
        for _ in range(markov_length):
            i = random.randint(1, len(route) - 1)
            j = random.randint(1, len(route) - 1)

            while j == i or j-i == 1 or j-i == -1:
                j = random.randint(1, len(route) - 1)
                
            current_temp = cooling_schedule(start_temp, max_iterations, iteration, cooling)
            cost = cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j])
            ap = acceptance_probability(cost, current_temp)

            # check if next state to be accepted
            if random.uniform(0, 1) < ap:
                best[i:j] = best[j - 1:i - 1:-1]
                distance += cost

            # Save 1 in 100 steps for plot
            if counter % 100 == 0:
                costs.append(calculate_cost(best, cost_mat))
                count += 1
                iter2.append(count)
                cooling2.append(cooling)

            counter += 1

        route = best
    return best, costs, iter2, cooling2


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
    # markov_length = [10, 25, 50, 75, 100, 125, 150]
    markov_length = [100]
    
    costs = []
    temperatures_v = []
    schedules = []
    percentage_v = []
    init_routes = []
    init_costs = []
    routes = []
    markovs = []
    cost_during_run = []
    iter2 = []
    
    print(cooling)
    temps = temperatures[cooling]

    for markov in markov_length:
        print(cooling, markov)
        for i in range(len(temps)):
            temp = temps[i]
            percentage = percentages[i]

            max_i = 301
            for i in range(1, max_i):
                if i%50==0:
                    print("i:", i)
                nn_route = nearest_neighbour(matrix)
                distance_nn = calculate_cost(nn_route, matrix)
                
                result = sa_two_opt(nn_route, cost_mat, distance_nn, cooling, iteration, temp, markov)
                best_route = result[0]
                cost_over_sim = result[1]
                iter22 = result[2]
                cooling2 = result[3]

                cost_during_run.extend(cost_over_sim)
                iter2.extend(iter22)                
                schedules.extend(cooling2)
    
    data = {"Cooling_schedule":schedules, "Cost in run":cost_during_run, "Iter2":iter2}
    df = pd.DataFrame(data) 
    df
    df.to_csv(f"data/values_{cooling}_{max_i}_iter{markov_length[0]}ml.csv")


