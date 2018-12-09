import numpy as np
import matplotlib.pyplot as plt

def main():
    arr_va = np.loadtxt(open("va_file_data_50_reps_synthetic_data.csv", "rb"), delimiter=",")
    arr_rtree = np.loadtxt(open("rtree_data_50_reps_synthetic_data.csv", "rb"), delimiter=",")
    # graphing will be explored in different file
    y_data_linear = np.full((15, ), 100, dtype=int)
    x_data = np.arange(3, 48, 3)
    y_data_va = arr_va[:, [1]].flatten()
    # assuming that the data is not a percentage, but a fraction
    y_data_rtree = arr_rtree[:, [1]].flatten() * 100
    plt.plot(x_data, y_data_linear, '-b', marker='|', label='linear')
    plt.plot(x_data, y_data_va, '-r', marker='o', label='vafile')
    plt.plot(x_data, y_data_rtree, '-g', marker='o', label='rtree')
    plt.ylim(0,150)
    plt.legend(loc='upper left')
    plt.ylabel('% Vector/Leaf blocks visited')
    plt.xlabel('Number of dimensions in vectors')
    plt.title('N=50,000 uniformly distributed, k=10')
    #plt.title('N=50,000 image dataset, k=10')
    plt.show()

if __name__ == '__main__':
    main()
