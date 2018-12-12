import numpy as np
from utils import knn
from vafile import *

def main():
    data = np.random.rand(50000, 2)
    q = data[22000]
    va_inst = VAFile(data, 8)
    va_inst.near_optimal_search(q, 20)
    print ("Return for 20 nearest neighbors to q on a 50000 by 2 random data array")
    _ , dists = knn(data, q, 20)
    print (dists)
    print ("Sorting va_inst.dst")
    sorted_indexes = np.argsort(va_inst.dst)
    print ("Returning top 20 results from dst array")
    nn_indexes = sorted_indexes[:20]
    nn_dists = va_inst.dst[nn_indexes]
    print (nn_dists)



if __name__ == '__main__':
    main()