import numpy as np
from rtree import RTree
from node import Node 



tree = RTree(10,5)
tree.insert([1,2,3,4,5],420)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([6,6,6,6,6],69)
tree.insert([7,7,7,7,7],69)
print(tree.root.BBs)
print(tree.root.objects)
print(tree.root.size)