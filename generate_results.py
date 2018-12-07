import numpy as np
from va_file.vafile import VAFile
import math
import matplotlib.pyplot as plt
from random import randint

# read in the the feature vector txt file as numpy array
percentage_list = []
for dim_index in range(3, 48, 3):
    # load incremental feature vector data
    a = np.loadtxt('./data/fv/fv' + str(dim_index) + 'd.txt')
    # create a VAFIle inst
    va_inst = VAFile(a, 8)

    rand_index = randint(0, 50000)
    q = a[rand_index]
    #print a[0].shape
    return_val = va_inst.near_optimal_search(q, 10)
    number_of_vectors_per_block = math.floor(8000/dim_index)
    number_of_blocks = math.floor(50000/number_of_vectors_per_block)
    print ("Now I am doing dimension {}, the return val is {} and the block number is {} and the percentage is {}".format(dim_index, return_val, number_of_blocks, round(return_val/number_of_blocks * 100, 1)))
    percentage_list.append(round(return_val/number_of_blocks * 100, 1))
    #print percentage_list


y_data_linear = np.full((15, ), 100, dtype=int)
x_data = np.arange(3, 48, 3)
y_data_va = np.asarray(percentage_list)
plt.plot(x_data, y_data_linear, '-b', marker='|', label='linear')
plt.plot(x_data, y_data_va, '-r', marker='o', label='vafile')
plt.ylim(0,140)
plt.legend(loc='upper left')
plt.ylabel('% of Vector blocks visited')
plt.xlabel('Number of dimensions in vectors')
plt.show()
