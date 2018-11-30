import numpy as np 
from bb_utils import bb_area, bb_list_merge

class Node:
    def __init__(self, parent, is_leaf, max_size, data_dim, tree=None):
        self.is_leaf = is_leaf
        self.max_size = max_size
        self.data_dim = data_dim
        self.size = 0
        self.tree = tree
        
        self.parent = parent
        # this node's bounding box
        self.mbb = np.zeros((2, data_dim))
        # the list of children of this node
        # this is list of (mbb, rid) tuples for leaf nodes
        self.objects = [0]*max_size

    # update the current node's bounding box to reflect the current data
    # also works on Points :)
    def update_bb(self):
            self.mbb = bb_list_merge([o.mbb for o in self.objects[:self.size]])


    # make parent's bbs compliant with children
    def fix_up(self):
        curr = self
        while curr.parent:
            curr.parent.update_bb()
            curr = curr.parent

    # finds the child with minimally expanded bbox
    def minimal_expand(self, mbb):
        min_diff = np.inf
        best_child = None
        best_bbox = None
        for child in self.objects[:self.size]:
            BB = child.mbb
            area_before = bb_area(BB)
            new_bbox = bb_list_merge([BB,mbb])
            area_after = bb_area(new_bbox)
            diff = area_after - area_before
            if diff < min_diff:
                min_diff = diff
                best_child = child
                best_bbox = new_bbox
        return best_child, new_bbox

    # obj is an rid for leaves, a node for internal nodes
    def add(self, obj):
        assert self.size < self.max_size
        # add object, expand box, increment size
        self.objects[self.size] = obj
        self.size += 1
        obj.parent = self
        self.update_bb()

    # inserts point into node, splits node and propagates the split
    def insert(self, obj):
        if self.size < self.max_size:
            self.add(obj)


        elif self.size == self.max_size:


            new_node = Node(self.parent, self.is_leaf, self.max_size, self.data_dim, self.tree)

            # "remove" data from original node
            self.size = self.max_size//2
            self.update_bb()
            

            # add data to new node
            new_node.objects[:new_node.max_size//2] = self.objects[self.max_size//2:]
            new_node.size = self.max_size//2
            new_node.add(obj)


            # if the node we split was not the root, insert new node into the parent
            if self.parent:
                self.parent.insert(new_node)


            # if the node we split was the root, create a new root
            elif self.parent == None:
                new_root = Node(None, False, self.max_size, self.data_dim, self.tree)
                # the root's parent is the tree obj
                # set the tree's root to be the new root
                self.tree.root = new_root
                new_root.add(self)
                new_root.add(new_node)

            else:
                print('should never get here, parent is not None and is not not None')
                assert(False)

            # fix boxes after everything is inserted and the dust settles
            self.fix_up()

        else:
            print('should never get here, size > max_size')
            assert(False)


    def __str__(self):
        variant = 'leaf' if self.is_leaf else 'internal'
        print(variant, 'node')
        print('bounding box:', self.mbb)
        print(self.size, 'objects:')
        for o in self.objects[:self.size]:
            print(o)
        return ''














