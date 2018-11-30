import numpy as np 
from node import Node
from point import Point
from bb_utils import bb_contain

class RTree:
    def __init__(self, elements_per_node, data_dim):
        self.root = Node(None, is_leaf=True, max_size=elements_per_node, data_dim=data_dim, tree=self)

    def insert(self, coords, rid):
        p = Point(coords, rid)
        self.insert_point(self.root, p)


    @staticmethod
    def insert_point(node, p):
        if not node.is_leaf:
            for child in node.objects[:node.size]:
                if bb_contain(child.mbb, p.mbb):
                    RTree.insert_point(child, p)
                    return
            # gets here if there was no box to put the point in
            # find the child of node which neadsto expand the least to hold mbb
            best_child, new_bb = node.minimal_expand(p.mbb)
            RTree.insert_point(best_child, p)
            best_child.mbb = new_bb

        elif node.is_leaf:
            node.insert(p)
            








