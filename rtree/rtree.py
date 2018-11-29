import numpy as np 
from node import Node
from bb_utils import bb_contain

class RTree:
    def __init__(self, elements_per_node, data_dim):
        self.root = Node(parent=None, is_leaf=True, max_size=elements_per_node, data_dim=data_dim)

    def insert(self, mbb, rid):
        self.insert_point(self.root, mbb, rid)


    def insert_point(self, node, mbb, rid):
        if not node.is_leaf:
            for BB, child in node:
                if bb_contain(BB, mbb):
                    insert_point(child, mbb, rid)
                    return
            # gets here if there was no box to put the point in
            # find the child of node which neadsto expand the least to hold mbb
            best_child = node.minimal_expand(mbb)
            insert_point(best_child, mbb, rid)

        elif node.is_leaf:
            if node.size < node.max_size:
                node.insert(mbb, rid)

            # will have to make a new node in order to insert
            else:
                node.insert_split(self, mbb, rid)






