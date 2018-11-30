import numpy as np
from rtree import RTree
from node import Node
import time


start_time = time.time()

n = 50000
dim = 10
per_page = 8000//(10*2*4 + 4)
print(per_page, 'entries per page')

tree = RTree(per_page,dim)

for i in range(50000):
    tree.insert(np.random.rand(dim),i)

print(time.time() - start_time, 'seconds')


#print(tree.root)


