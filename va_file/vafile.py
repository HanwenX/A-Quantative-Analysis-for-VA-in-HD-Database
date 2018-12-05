import numpy as np
import heapq

class VAFile:
    """
    Details in the paper <An Approximation-Based Data Structure for Similarity Search>
    """
    def __init__(self, features, b_i):
        """
        Input:
        features - N*d numpy array
        b_i - number of bit along each dimension

        Intialize a feature approximation vector array of size N*d as an 
        uint8 numpy array where each entry is an unpacked representation
        of 8 bits.
        The edges of bins along each dimension is held in markers
        """
        # get dimensions from the features      
        N, d = features.shape
        b = b_i * d

        # partition the features into 2**b_i bins along each dimension
        # each bin contains rougly the same amount of points
        sorted_features = np.sort(features, axis=0)
        bins = 2**b_i
        pts_per_bin = int(np.floor(N/bins))
        markers = sorted_features[pts_per_bin:pts_per_bin*bins:pts_per_bin, :].T
        markers = np.hstack((np.zeros((d, 1)),markers, np.ones((d, 1))))
        va = np.zeros((N, d), dtype=np.uint8)

        # vectorized version of generating vector approximation
        for dim in xrange(d):
            assigned = np.zeros((N), dtype=bool)
            for r in xrange(1, bins+1):
                fall_in =  features[:, dim] < markers[dim, r]
                va[np.logical_and(fall_in, np.logical_not(assigned)), dim] = r-1
                assigned = np.logical_or(assigned, fall_in)

        self.features = features.copy()
        self.va = va
        self.markers = markers

#    def init_candidate(feature_approximation):
#        # this function returns nth largest upper bound encountered so far
#        maxint = 0
#        N, d = feature_approximation.shape
#        for k in range(N):
#            if(feauture_approximation[k] >= maxint):
#                maxint = feauture_approximation[k]
#        return maxint

    def init_candidate(self, k):
        """
        Initialize distance array to infinity as upper bounds and return infinity        
        Array ans holds the index for the k nearest candidates
        """
        self.dst = np.full(k, np.inf)
        self.ans = np.full(k, np.nan)
        return np.inf        

    def candidate(self, delta, i):
        """
        Update the k nearest candidates and return the largest upper bound so far
        """
        if delta < self.dst[-1]:
            self.dst[-1] = delta
            self.ans[-1] = i
            ind = np.argsort(self.dst)
            self.dst = self.dst[ind]
            self.ans = self.ans[ind]

        return self.dst[-1]

    def weighted_dist(self, a, b):
        """
        This is the weighted metric L_p(a, b) defined as in section 3
        """
        # p = 2 for L2 distance
        p = 2.0
        # we use weight of all ones
        weight = np.ones_like(a, dtype=np.float64)
        dist = np.power(np.sum((weight*np.fabs(a-b))**p), 1/p)

        return dist

    def get_bounds(self, a, q):
        """
        Return lower and upper bounds based on the formula provided in the paper
        """
        # p = 2 for L2 distance
        p = 2.0
        # get the vector approximation of q and use the same weight as in weighted_dist
        a_q = self.approximate_vector(q)      
        weight = np.ones_like(a, dtype=np.float64)
        # initialize upper and lower bound
        l = np.zeros_like(a, dtype=np.float64)
        u = np.zeros_like(a, dtype=np.float64)
        # index for three conditions
        less = (a < a_q)
        bigger = (a > a_q)
        equal = (a == a_q)
        # compute lower bound
        l[less] = q[less] - self.markers[less, a[less] + 1]
        l[bigger] = self.markers[bigger, a[bigger]] - q[bigger]
        l_i = np.power(np.sum((weight*l)**p), 1/p)
        # compute upper bound
        u[less] = q[less] - self.markers[less, a[less]]
        u[equal] = np.fmax(q[equal] - self.markers[equal, a[equal]], self.markers[equal, a[equal] + 1] - q[equal])
        u[bigger] = self.markers[bigger, a[bigger] + 1] - q[bigger]
        u_i = np.power(np.sum((weight*u)**p), 1/p)

#        if l_i > u_i:
#            print 'exception!'
#            print l_i, u_i            

        return (l_i, u_i)

    def near_optimal_search(self, q, k):
        """
        Get the k nearest neighbors to the query q.
        But we only return the count of blocks visied for this project.
        """
        N, _ = self.features.shape
        hp = []
        # phase 1
        delta = self.init_candidate(k)
        for i in xrange(N):
            l_i, u_i = self.get_bounds(self.va[i, :], q)
            if l_i <= delta:
        #        print delta
                delta = self.candidate(u_i, i)
                heapq.heappush(hp, (l_i, i))
        print 'finish phase 1'
#        print self.dst, len(hp)

        # phase 2
        delta = self.init_candidate(k)        
        l_i, i = heapq.heappop(hp)
        count = 0
        while l_i < delta and hp:
#            print l_i, delta
            delta = self.candidate(self.weighted_dist(self.features[i, :], q), i)
            count += 1            
            l_i, i = heapq.heappop(hp)
        print 'finish phase 2'
        return count

    def approximate_vector(self, q):
        """
        Return the vector approximation of vector q
        """
        a = np.zeros_like(q, dtype=np.uint8)
        for dim in xrange(self.markers.shape[0]):
            for r in xrange(1, self.markers.shape[1]+1):
                if q[dim] < self.markers[dim, r]:
                    a[dim] = r-1
                    break
        return a

    def get_features(self):
        return self.features

    def get_va(self):
        return self.va

    def get_markers(self):
        return self.markers

