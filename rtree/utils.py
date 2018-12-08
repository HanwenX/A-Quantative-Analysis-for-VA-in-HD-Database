import numpy as np


def knn(data, q, k):
    dists = np.sqrt( np.sum( np.power(data-q,2) , axis=1) )
    sorted_indexes = np.argsort(dists)
    nn_indexes = sorted_indexes[:k]
    nn_dists = dists[nn_indexes]
    return nn_indexes, nn_dists
