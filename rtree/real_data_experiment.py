import numpy as np
from rtree import RTree
from utils import knn


def main():
    n = 50000

    max_dim = 45
    dims = np.arange(2,max_dim+1, step=2)
    

    # dims = [2,5,10,15,20,25,30,35,40,45]

    reps = 50
    m = 0.4
    p = 0.3
    k = 10

    page_size = 8000
    num_bytes = 8
    dim_percent = {d:[] for d in dims}

    for dim in dims:
        print('testing dimension:', dim)
        M = np.floor(page_size/(2*dim*num_bytes + num_bytes)).astype(int)

        tree = RTree(M, dim, m, p)
        #data = np.random.rand(n,dim)
        data = np.loadtxt('../data/fv/fv'+str(dim)+'d.txt')


        data = data[:n]
        #print(data.shape)

        tree.insert_batch(data, np.arange(n))

        for i in np.arange(reps):
            #q = np.random.rand(dim)
            q_idx = np.random.randint(0,n)
            q = data[q_idx,:]

            rtree_nns, num_visited_leaves = tree.knn_naive(q,k)
            # true_nns, true_dists = knn(data, q, k)
            # rtree_nns = [x.rid for x in rtree_nns]
            # print('rtree nns:', rtree_nns)
            # print('true nns:', true_nns)
            leaves = tree.get_leaves()
            total_leaves = len(leaves)
            dim_percent[dim].append(num_visited_leaves/total_leaves)

        print('dim:', dim, '%:', np.mean(dim_percent[dim])*100)

    dim_percent = {k:np.mean(v) for k,v in dim_percent.items()}

    percents = [dim_percent[d] for d in dim_percent.keys()]

    out = np.vstack([dims, percents]).T 
    np.savetxt('data/rtree_real.csv', out, delimiter=',')













if __name__ == '__main__':
    main()
