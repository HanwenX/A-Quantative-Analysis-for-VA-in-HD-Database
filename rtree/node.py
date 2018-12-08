import numpy as np 
from bb_utils import bb_area, bb_margin, bb_list_merge, bb_overlap
import time

class Node:
    def __init__(self, parent, is_leaf, max_size, data_dim, min_frac, tree=None):
        self.is_leaf = is_leaf
        self.max_size = max_size
        self.data_dim = data_dim
        self.size = 0
        self.tree = tree
        assert(self.tree)
        self.is_node = True
        self.min_frac = min_frac
        self.min_size = int(min_frac*max_size)
        self.p = tree.p
        self.parent = parent
        if self.parent == None:
            self.level = 0
        else:
            self.level = self.parent.level + 1
        # this node's bounding box
        self.mbb = np.zeros((2, data_dim))
        # the list of children of this node
        # this is list of (mbb, rid) tuples for leaf nodes
        self.objects = []

        #self.split = self.dumbass_split
        self.split = self.rstar_split

    # update the current node's bounding box to reflect the current data
    # also works on Points :)
    def update_bb(self):
            self.mbb = bb_list_merge([o.mbb for o in self.objects[:self.size]])


    # shrink all parent bb's after a split
    def fix_up(self):
        curr = self
        while curr.parent:
            curr.parent.update_bb()
            curr = curr.parent

    # finds the child with minimally expanded bbox
    def minimal_expand(self, mbb):
        min_diff = np.inf
        min_area = np.inf
        best_child = None
        best_bbox = None
        for child in self.objects[:self.size]:
            BB = child.mbb
            area_before = bb_area(BB)
            new_bbox = bb_list_merge([BB,mbb])
            area_after = bb_area(new_bbox)
            diff = area_after - area_before
            if (diff <= min_diff) and (area_before < min_area):
                min_diff = diff
                min_area = area_before
                best_child = child
                best_bbox = new_bbox
        return best_child

    # adds obj to self node
    # obj is an rid for leaves, a node for internal nodes
    def add(self, obj):
        assert self.size < self.max_size
        # add object, expand box, increment size
        self.objects.append(obj)
        self.size += 1
        obj.parent = self
        self.update_bb()

    # inserts point into node, splits node and propagates the split
    def insert(self, obj):
        if self.size < self.max_size:
            self.add(obj)

        elif self.size == self.max_size:
            # reinsert operation
            if self.level >= self.tree.reinsert_level and self.is_leaf:
                self.forced_reinsert(obj)

            # normal split insert
            else:
                new_node = Node(self.parent, self.is_leaf, self.max_size, self.data_dim, self.min_frac, self.tree)

                # split data, put half in original node, half in new node
                left, right = self.split(obj)

                self.objects_replace(left)
                new_node.objects_replace(right)

                # if the node we split was not the root, insert new node into the parent
                if self.parent:
                    self.parent.insert(new_node)

                # if the node we split was the root, create a new root
                elif self.parent == None:
                    new_root = Node(None, False, self.max_size, self.data_dim, self.min_frac, self.tree)
                    # the root's parent is the tree obj
                    # set the tree's root to be the new root
                    self.tree.root = new_root
                    new_root.add(self)
                    new_root.add(new_node)
                    new_root.update_levels()
                else:
                    print('should never get here, parent is not None and is not not None')
                    assert(False)
                # fix boxes after everything is inserted and the dust settles
                self.fix_up()
        else:
            print('should never get here, size > max_size')
            assert(False)

    def objects_replace(self, objects):
        self.objects = objects
        self.size = len(objects)
        self.update_bb()

    def dumbass_split(self, obj):
        left = [*self.objects[:self.max_size//2], obj]
        right = self.objects[self.max_size//2:]
        return left, right

    def linear_split(self, obj):
        pass

    def quadratic_split(self, obj):
        pass

    def rstar_split(self, obj):
        objs = [*self.objects[:self.size], obj]
        #o.mbb is 2xd, so this is M+1x2xd
        mbbs = np.array([o.mbb for o in objs])
 
        best_axis = None
        best_S = np.inf
        # choose split axis:
        for axis in range(self.data_dim):
            # two different ways of sorting, lexsort faster when more things per page
            data = mbbs[np.lexsort(np.flip(mbbs[:,:,axis],axis=1).T, axis=0)]
            #data = np.array(sorted(mbbs, key=lambda x : (x[0,axis], x[1,axis])))
            S = 0.0
            for k in range(self.max_size - 2*self.min_size):
                left = data[:self.min_size+k,:,:]
                right = data[self.min_size+k:,:,:]
                left_bb = bb_list_merge(left)
                right_bb = bb_list_merge(right)
                S += bb_margin(left_bb) + bb_margin(right_bb)

            if S < best_S:
                best_S = S
                best_axis = axis

        best_k = None
        best_overlap = np.inf
        data_indices = np.lexsort(np.flip(mbbs[:,:,best_axis],axis=1).T, axis=0)
        data = mbbs[data_indices]

        # print('best_axis:', best_axis)
        # print('data:', data)

        for k in range(self.max_size - 2*self.min_size + 2):
            left = data[:self.min_size+k,:,:]
            right = data[self.min_size+k:,:,:]
            left_bb = bb_list_merge(left)
            right_bb = bb_list_merge(right)
            overlap = bb_overlap(left_bb, right_bb)
            if overlap < best_overlap:
                best_overlap = overlap
                best_k = k

        left_indices = data_indices[:self.min_size+best_k]
        right_indices = data_indices[self.min_size+best_k:]

        left = [objs[i] for i in left_indices]
        right = [objs[i] for i in right_indices]

        return left, right

    def forced_reinsert(self, obj):
        print('forced reinserting!')
        assert(self.is_leaf)
        objs = [*self.objects[:self.size], obj]
        #o.mbb is 2xd, so this is M+1xd
        # this is a leaf node so all objects are points with same upper and lower points
        points = np.array([o.mbb[0] for o in objs])
        centroid = np.mean(points, axis=0)
        k = np.round((self.max_size+1)*(1-self.p),0).astype(int)

        dists = np.sqrt( np.sum( np.power(points - centroid,2), axis=1 ) )
        indices = np.argsort(dists)

        self.objects = [objs[i] for i in indices[:k]]
        self.size = len(self.objects)
        self.update_bb()
        self.fix_up()
        self.tree.reinsert_level += 1

        for i in range(k+1,self.max_size+1):
            self.tree.insert_point(self.tree.root, objs[i])

    # updates node levels after a new root is created
    def update_levels(self, level=0):
        self.level = level
        if self.is_leaf:
            return
        else:
            for obj in self.objects:
                obj.update_levels(level+1)
            return


    def __str__(self):
        variant = 'leaf' if self.is_leaf else 'internal'
        print(variant, 'node')
        print('bounding box:', self.mbb)
        print(self.size, 'objects:')
        for o in self.objects[:self.size]:
            print(o)
        return ''














