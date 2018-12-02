import numpy as np
from rtree import RTree


def main():
    n = 50000
    max_dim = 30
    dims = np.arange(2,max_dim+1, step=1)

    #dims = [1,2,4,6,8,10,12,15,20,25,30]

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
        data = np.random.rand(n,dim)
        tree.insert_batch(data, np.arange(n))

        for i in np.arange(reps):
            q = np.random.rand(dim)
            _, num_visited_leaves = tree.knn_naive(q,k)
            leaves = tree.get_leaves()
            total_leaves = len(leaves)
            dim_percent[dim].append(num_visited_leaves/total_leaves)

    dim_percent = {k:np.mean(v) for k,v in dim_percent.items()}

    percents = [dim_percent[d] for d in dim_percent.keys()]

    out = np.vstack([dims, percents]).T 
    np.savetxt('data/rtree_data.csv', out, delimiter=',')













if __name__ == '__main__':
    main()
