
import matplotlib.pyplot as plt
def plot(route, version):    
    cities = {}
    tsp_file = "a280"

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

    for i in range(len(route) - 1):
        city1 = cities[route[i]]
        city2 = cities[route[i+1]]
        plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    city1 = cities[route[-1]]
    city2 = cities[route[0]]
    plt.plot([city1[0], city2[0]], [city1[1], city2[1]])

    plt.savefig(f'{version}_route.png')