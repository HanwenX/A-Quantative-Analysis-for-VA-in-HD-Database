import numpy as np
from utils import cluster_data
import os


def main():
    if not os.path.exists('data/clusters'):
        os.mkdir('data/clusters')

    for dim in range(2,31):
        data = cluster_data(50000, dim, 10, 0.01)
        print(data.shape)
        np.save('data/clusters/clusters_' + str(dim) + 'd', data)


if __name__ == '__main__':
    main()