import matplotlib.pyplot as plt 
import numpy as np
from operator import itemgetter
import copy
import random
import math
import pandas as pd
import sys

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
        # additive
        # current_temp = start_temp * ((max_iterations-iteration) / max_iterations)
        # multiplicative
        alpha = 1
        current_temp = start_temp/(1 + alpha*iteration)
    elif kind == "Log":
        alpha = 50
        # multiplicative
        current_temp =  start_temp/(alpha * (math.log(iteration + 1, 10)))
    elif kind == "Exponential":
        # additive 
        unit = 1 + np.exp((2 * np.log(start_temp)/max_iterations) * (iteration - (0.5 * max_iterations)))
        current_temp = start_temp * (1 / unit)
    elif kind == "Quadratic":
        # additive
        current_temp = start_temp * ((max_iterations - iteration)/max_iterations)**2
    
    return current_temp

def sa_two_opt(route, cost_mat, distance, cooling, max_iterations, start_temp):
    best = route
    prev_distance = 0
    iteration = 0
    counter = 0

    while iteration < max_iterations:
        iteration += 1
        count = 0
        
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: 
                    continue

                count += 1
                
                current_temp = cooling_schedule(start_temp, max_iterations, iteration, cooling)
                cost = cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j])
                ap = acceptance_probability(cost, current_temp)

                if random.uniform(0, 1) < ap:
                    best[i:j] = best[j - 1:i - 1:-1]
                    distance += cost

                if count >= 1000:
                    break
            if count >= 1000:
                break
        # print(count)

        # distance = round(distance, 5)
        # if prev_distance == distance:
        #     counter += 1
        #     # print("hoi", counter, iteration)
        #     if counter >= 100:
        #         print("klaar", iteration, distance)
        #         return best
        # else:
        #     counter = 0

        prev_distance = distance

        route = best
    return best


if __name__ == '__main__':
    tsp_file = "a280"

    matrix = make_matrix(tsp_file)
    cost_mat = list(matrix)

    cooling = sys.argv[1]

    temperatures = {}
    # temperatures["Exponential"] = [150, 220, 450]
    # temperatures["Linear"] = [530, 850, 1700]
    # temperatures["Log"] = [120, 180, 370]
    # temperatures["Quadratic"] = [530, 850, 1700]

    temperatures["Exponential"] = [200, 200, 200, 200, 200, 200]
    temperatures["Linear"] = [230, 200, 200, 200, 200, 200]
    temperatures["Log"] = [3000, 3000, 3000, 3000, 3000, 3000]
    temperatures["Quadratic"] = [250, 200, 200, 200, 200, 200]

    iterations = [10000]
    # iterations = [10, 50, 100]
    
    costs= []
    temperatures_v=[]
    schedules = []
    iteration_v = []
    routes = []
    
    print(cooling)
    temps = temperatures[cooling]

    max_i = 2
    for i in range(1, max_i):
        # if i%20 == 0:
            # print(f"progress out of {max_i}:", i)
        # amount of iterations per simulation
        for count, iteration in enumerate(iterations):
            temp = temps[count]
            nn_route = nearest_neighbour(matrix)
            distance_nn = calculate_cost(nn_route, matrix)
            print("nn distance:", distance_nn)

            result = sa_two_opt(nn_route, cost_mat, distance_nn, cooling, iteration, temp)
            best_route = result
            print(f"after {iteration} iterations, na sa cost:", calculate_cost(best_route, matrix))
            
            cost = calculate_cost(best_route, matrix)
            costs.append(cost)
            iteration_v.append(iteration)
            schedules.append(cooling)
            temperatures_v.append(temp)
            routes.append(best_route)


    data = {'Iterations':iteration_v, "Cooling_schedule":schedules, "Cost":costs, "Initial temperature":temperatures_v, "Routes":routes}
    df = pd.DataFrame(data) 
    df
    df.to_csv(f"data/values_{cooling}_{iterations[-1]}_iter.csv")
    
  
        



    # # start plot
    # # Initializes lists and dictionaries
  
     