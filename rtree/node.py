import numpy as np 
from bb_utils import bb_area, bb_merge, bb_list_merge

class Node:
    def __init__(self, parent, is_leaf, max_size, data_dim):
        self.parent = parent
        self.is_leaf = is_leaf
        self.max_size = max_size
        self.data_dim = data_dim
        self.size = 0
        # first dimension is index into node's bounding box array
        # second dimension is what data dimension, 3rd dimension is lower left/top right
        self.BBs = np.zeros((max_size, data_dim, 2))
        self.objects = [0]*max_size

    # this function allows me to iterate over a node
    def __iter__(self):
        return zip(self.BBs[:self.size], self.objects[:self.size])

    # finds and returns index ofthe child with minimally expanded bbox
    # also gives that child
    def minimal_expand(self, mbb):
        min_diff = np.inf
        best_idx = None
        best_bbox = None
        for idx in range(self.size):
            BB = self.BBs[idx,:,:]
            child = self.objects[idx]
            area_before = bb_area(BB)
            new_bbox = bb_merge(BB,mbb)
            area_after = bb_area(new_bbox)
            diff = area_after - area_before
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
                best_bbox = new_bbox
        # update the bbox
        self.BBs[best_idx,:,:] = best_bbox
        return self.objects[best_idx]


    def insert(self, mbb, rid):
        assert self.size < self.max_size
        self.BBs[self.size,:,:] = np.vstack([mbb,mbb]).T
        self.objects[self.size] = rid
        self.size += 1

    # inserts point into node, splits node and propagates the split
    def insert_split(self, tree, mbb, rid):
        # case where root is a leaf
        if self.parent == None:
            new_root = Node(None, is_leaf=False, max_size=self.max_size, data_dim=self.data_dim)
            new_leaf = Node(None, is_leaf=True, max_size=self.max_size, data_dim=self.data_dim)
            
            # "remove" data from original node
            self.size = self.max_size//2

            # add data to new node
            new_leaf.BBs[:self.max_size//2,:,:] = self.BBs[self.max_size//2:,:,:]
            new_leaf.BBs[self.max_size//2-1,:,:] = np.vstack([mbb,mbb]).T
            new_leaf.objects[:self.max_size//2] = self.objects[self.max_size//2:]
            new_leaf.objects[self.max_size//2-1] = rid
            new_leaf.size = self.max_size/2

            new_root.BBs[0,:,:] = bb_list_merge(self.BBs[:self.size,:,:])
            new_root.BBs[1,:,:] = bb_list_merge(new_leaf.BBs[:self.size,:,:])

            new_root.objects[:2] = [self, new_leaf]
            new_root.size = 2

            self.parent = new_root
            new_leaf.parent = new_root
            
            tree.root = new_root




