import numpy as np
from M import *

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

def sa_two_opt(route, cost_mat):
    start_temp = 10
    max_iterations = 1000
    accepted = 0

    best = route

    for iteration in range(max_iterations):
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue

                current_temp = start_temp - (start_temp/max_iterations) * iteration
                cost = cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j])
                ap = acceptance_probability(cost, current_temp)
                
                if random.uniform(0, 1) < ap:
                    best[i:j] = best[j - 1:i - 1:-1]
                    accepted += 1
        route = best
    return best

def calculate_cost(order, matrix):
    distance = 0

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

if __name__ == '__main__':
    tsp_file = "a280"

    matrix = make_matrix(tsp_file)
    
    nodes = len(matrix)
    result = nearest_neighbour(matrix)
    print("nn distance:", result[0])
    
    nn_route = result[1]
    print("len nn", len(nn_route))

    cost_mat = list(matrix)
    # best_route = sa_two_opt(nn_route, cost_mat)
    # print(best_route)
    
    best_route = [74, 76, 75, 73, 72, 71, 70, 69, 66, 68, 67, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 58, 62, 63, 64, 65, 84, 85, 115, 112, 86, 83, 82, 87, 111, 108, 107, 109, 110, 113, 114, 116, 117, 61, 60, 59, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 21, 24, 22, 23, 13, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 276, 275, 274, 273, 272, 271, 270, 15, 16, 17, 132, 131, 18, 19, 130, 129, 20, 128, 127, 126, 125, 124, 123, 122, 121, 120, 153, 154, 152, 155, 151, 119, 118, 156, 157, 158, 159, 174, 173, 172, 106, 105, 104, 103, 102, 101, 91, 90, 89, 88, 81, 80, 79, 78, 77, 94, 95, 96, 93, 92, 97, 98, 99, 100, 168, 169, 171, 170, 167, 166, 165, 164, 163, 162, 161, 160, 180, 175, 176, 150, 177, 149, 178, 179, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 200, 199, 201, 202, 203, 204, 205, 207, 252, 251, 208, 209, 206, 211, 210, 213, 212, 215, 214, 217, 216, 219, 220, 221, 218, 222, 223, 224, 225, 226, 227, 228, 229, 250, 249, 246, 244, 245, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 243, 242, 241, 1, 0, 279, 2, 278, 277, 247, 248, 255, 254, 253, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 133, 134, 135, 136, 137, 138, 148, 147, 139, 140, 141, 146, 145, 142, 143, 144, 198, 197, 196, 192]
    print("na sa cost:", calculate_cost(best_route, matrix))
    print("best route len:", len(best_route))

    best_2opt = two_opt(nn_route, cost_mat)
    print("hoi", best_2opt)
    print("2opt cost:", calculate_cost(best_2opt, cost_mat))