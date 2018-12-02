import numpy as np 
from node import Node
from point import Point
from bb_utils import bb_contain, dist, min_dist, max_dist

class RTree:
    def __init__(self, elements_per_node, data_dim, min_frac=0.4, p=0.3):
        self.p = p
        self.root = Node(None, is_leaf=True, max_size=elements_per_node, data_dim=data_dim, min_frac=min_frac, tree=self)
        self.reinsert_level = 1

    def insert_batch(self, data, rids):
        for coords, rid in zip(data,rids):
            self.insert(coords, rid)

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

    def knn_naive(self, q, k):
        neighbors = []
        nnk_dist = np.inf
        done = False
        leaves = self.get_leaves()
        leaves= sorted(leaves, key=lambda x : max_dist(x.mbb, q))
        num_leaves_visited = 0
        for leaf in leaves:
            num_leaves_visited += 1
            leaf_min_dist = min_dist(leaf.mbb, q)
            if leaf_min_dist > nnk_dist:
                break
            objs = sorted(leaf.objects[:leaf.size], key=lambda x : dist(x, q))
            neighbors.extend(objs[:k])
            neighbors = sorted(neighbors, key=lambda x : dist(x, q))[:k]
            nnk_dist = dist(neighbors[-1], q)
        return neighbors, num_leaves_visited

    def get_leaves(self):
        leaves = []
        self.get_leaves_helper(self.root,leaves)
        return leaves

    def get_leaves_helper(self, node, leaves):
        if node.is_node:
            if node.is_leaf:
                leaves.append(node)
            for n in node.objects[:node.size]:
                self.get_leaves_helper(n, leaves)
            











