import numpy as np
import random


def ortho(ns):
    assert(np.sqrt(ns) % 1 == 0),"Please insert an even number of samples"
    n = int(np.sqrt(ns))
    # Making a datastructure of a dict with coordinate tuples of a bigger grid with subcoordinate of sub-grid points
    blocks = {(i,j):[(a,b) for a in range(n) for b in range(n)] for i in range(n) for j in range(n)}
    points = []#np.empty((n,2))
    append = points.append # tips of python to fasten up append call
    for block in blocks:
        point = random.choice(blocks[block])
        lst_row = [(k1, b) for (k1, b), v in blocks.items() if k1 == block[0]]
        lst_col = [(a, k1) for (a, k1), v in blocks.items() if k1 == block[1]]

        for col in lst_col:
            blocks[col] = [a for a in blocks[col] if a[1] != point[1]]

        for row in lst_row:
            blocks[row] = [a for a in blocks[row] if a[0] != point[0]]

        #Adjust the points to fit the grid they fall in  
        point = (point[0] + n * block[0], point[1] + n * block[1])
        append(point)

    return points

def scale_points(points):
    x_l = []
    y_l = []
    p = ortho(points)
    maximum = points 
    scaling =[ 1/maximum * i for i in range(len(p))]
    min_ = 0
    max_ = 2
    result = np.zeros((points,2))
    anti_res = np.zeros((points,2)) # this is for antithetic variables

    for idx, scale in enumerate(scaling):
        x =  min_ + np.random.uniform(p[idx][0]/maximum, p[idx][0]/maximum +1/maximum ) * 3  # 4 is just max - min which is in my case 4
        y =  min_ + np.random.uniform(p[idx][1]/maximum, p[idx][1]/maximum + 1/maximum ) * 3
        result[idx, :] = [x,y]

    for i in result:
        x_l.append(i[0]-2)
        y_l.append(i[1]-1.5)
    
    x_l = np.array(x_l)
    y_l = np.array(y_l)

    return x_l, y_l

def scale_points_anti(points):
    x_l = []
    y_l = []
    x_a = []
    y_a = []
    p = ortho(points)
    maximum = points 
    scaling =[ 1/maximum * i for i in range(len(p))]
    min_ = 0
    max_ = 2
    result = np.zeros((points,2))
    anti_res = np.zeros((points,2)) # this is for antithetic variables

    for idx, scale in enumerate(scaling):
        x =  min_ + np.random.uniform(p[idx][0]/maximum, p[idx][0]/maximum +1/maximum ) * 3  # 4 is just max - min which is in my case 4
        y =  min_ + np.random.uniform(p[idx][1]/maximum, p[idx][1]/maximum + 1/maximum ) * 3
        result[idx, :] = [x,y]
        anti_res[idx,:] = [x*-1.0, y*-1.0] # antithetic variables

    for i in result:
        x_l.append(i[0]-2)
        y_l.append(i[1]-1.5)
        
        x_a.append(i[0]-2)
        y_a.append(i[1]-1.5)

    for i in anti_res:
        x_a.append(i[0]+1)
        y_a.append(i[1]+1.5)
    
    x_l = np.array(x_l)
    y_l = np.array(y_l)
    x_a = np.array(x_a)
    y_a = np.array(y_a)

    return x_l, y_l, x_a, y_a
