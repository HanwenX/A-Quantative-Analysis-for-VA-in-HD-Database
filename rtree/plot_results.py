import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



data = np.genfromtxt('rtree_data.csv', delimiter=',')

if len(data.shape) == 1:
    data = np.expand_dims(data,0)



dims = data[:,0]
percents = data[:,1]*100


plt.plot(dims,percents, 'r', label='R*-Tree')
plt.plot([0,*dims],np.zeros(len(dims)+1)+100, 'black', label='Scan')
plt.xlabel('Number of dimensions in vectors')
plt.ylabel('% Vector/Leaf blocks visited')
plt.title('N=50,000 uniformly distributed, k=10')
plt.legend()

plt.show()