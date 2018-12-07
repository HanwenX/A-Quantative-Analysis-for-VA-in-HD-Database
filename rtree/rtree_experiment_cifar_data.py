import numpy as np
from rtree import RTree
import math
import matplotlib.pyplot as plt
from random import randint


def main():
    n = 50000

    # 3 to 45 dimensions, step by 3
    dims = np.arange(45, 48, 3)

    reps = 20
    m = 0.4
    p = 0.3
    k = 10

    dim_percent = {d:[] for d in dims}

    for dim in dims:
        print('testing dimension:', dim)
        M = np.floor(8000/(2*dim*8 + 8)).astype(int)
        #M = 100
        tree = RTree(M, dim, m, p)
        data = np.loadtxt('../data/fv/fv' + str(dim) + 'd.txt')
        #data = np.random.rand(n,dim)
        tree.insert_batch(data, np.arange(n))

        for i in np.arange(reps):
            #q = np.random.rand(dim)
            # pass in the second row of dimension d as the query
            rand_index = randint(0, 50000)
            q = data[rand_index]

            _, num_visited_leaves = tree.knn_naive(q,k)
            leaves = tree.get_leaves()
            total_leaves = len(leaves)
            dim_percent[dim].append(num_visited_leaves/total_leaves)

    dim_percent = {k:np.mean(v) for k,v in dim_percent.items()}

    percents = [dim_percent[d] for d in dim_percent.keys()]

    out = np.vstack([dims, percents]).T
    print (out)
    #np.savetxt('data/rtree_data.csv', out, delimiter=',')

if __name__ == '__main__':
    main()
