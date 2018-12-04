import numpy as np

class VAFile:

    def partition(self, features, b_i):
        """
        Input:
        features - N*d numpy array
        b_i - number of bit along each dimension

        to_do:
        return features array of size N*b as uint8 numpy array
        return N * 2^(b_i + 1)** which holds the locations of the markers

        **I think it should be d * (2^b_i + 1) because we only need markers for each dimension
        my method won't include the 0s and 1s so it would be d * (2^b_i - 1)
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
        va = np.zeros((N, b), dtype=np.uint8)
        
        # naive version of generating vector approximation
#        for n in xrange(N):
#            for dim in xrange(d):
#                assigned = False
#                for r in xrange(bins-1):
#                    if features[n, dim] < markers[dim, r]:
#                        va[n, dim*b_i:(dim+1)*b_i] = np.unpackbits(np.uint8(r))
#                        break
#                    va[n, dim*b_i:(dim+1)*b_i] = np.unpackbits(np.uint8(bins-1))

        # vectorized version of generating vector approximation
        for dim in xrange(d):
            assigned = np.zeros((N), dtype=bool)
            for r in xrange(bins-1):
                fall_in =  features[:, dim] < markers[dim, r]
                va[np.logical_and(fall_in, np.logical_not(assigned)), dim*b_i:(dim+1)*b_i] = np.unpackbits(np.uint8(r))
                assigned = np.logical_or(assigned, fall_in)
            va[np.logical_not(assigned), dim*b_i:(dim+1)*b_i] = np.unpackbits(np.uint8(bins-1))

        return (va, markers)

    def init_candidate(feature_approximation):
        # this function returns nth largest upper bound encountered so far
        maxint = 0
        N, d = feature_approximation.shape
        for k in range(N):
            if(feauture_approximation[k] >= maxint):
                maxint = feauture_approximation[k]
        return maxint

    def candidate():
        """
        to_do:
        """
        pass

    def get_bounds(vector):
        """
        to_do:
        return lower and upper bounds based on the formula provided in the paper
        """
        pass

    def near_optimal_search(q):
        """
        Get the k nearest neighbors to the query q.
        """
        # phase 1
        #delta = init_candidate()
        #for i in xrange(N):
        #    l_i, u_i = get_bounds(a_i, q)
        #    if l_i <= delta:
        #        delta = candidate(u_i, i)
        #        insert_heap(heap, l_i, i)

        # phase 2
        #delta = init_candidate()
        #l_i, i = pop_heap(heap)
        #while l_i < delta:
        #    delta = candidate(L_p(p_i, q), i)
        #    l_i, i = pop_heap(heap)
        pass

