import numpy as np

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


if __name__ == '__main__':
    nodes = 51
    init_route = list(range(1, nodes))
    print(init_route)
    cost_mat = np.random.randint(100, size=(nodes, nodes))
    cost_mat += cost_mat.T
    np.fill_diagonal(cost_mat, 0)
    cost_mat = list(cost_mat)
    best_route = two_opt(init_route, cost_mat)
    print(best_route)