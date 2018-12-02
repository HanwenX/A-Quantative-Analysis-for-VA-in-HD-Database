import numpy as np
from rtree import RTree
from node import Node
import time
from bb_utils import dist, min_dist


start_time = time.time()

n = 50000
dim = 8
per_page = 8000//(dim*2*4 + 4)
# per_page = 20
print(per_page, 'entries per page')

tree = RTree(per_page,dim)

for i in range(n):
    tree.insert(np.random.rand(dim),i)
    if (i % (n/10)) == 0:
        print(int((i/n) *100), '%')

print('construct time:', time.time() - start_time, 'seconds')

leaves = tree.get_leaves()
lower_mean = np.mean([l.mbb[0] for l in leaves], axis=0)
upper_mean = np.mean([l.mbb[1] for l in leaves], axis=0)
print('lower mean:', lower_mean, 'upper mean:', upper_mean)

start_time = time.time()

total_leaves = len(tree.get_leaves())
q = np.random.rand(dim)
neighbors, num_leaves = tree.knn_naive(q,10)
print('search time:', time.time() - start_time, 'seconds')


print('leaves visited:', num_leaves, 'out of', total_leaves, (num_leaves/total_leaves)*100, '%')
for n in neighbors:
    print(n, dist(n, q))










