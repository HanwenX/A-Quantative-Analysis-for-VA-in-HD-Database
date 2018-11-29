import numpy

class VaFile:

    def partition(features, b):
        # input:
            # features ==> N*d numpy array
            # b ==> number of bits
        # get dimensions from the features
        N, d = features.size
        # b = b_sub_i * d
        b_sub_i = b / d
        """
        to_do:
        return features array of size N*b_sub_id as boolean numpy array
        return N * 2^(b_sub_i + 1) which holds the locations of the markers
        """
        return True

    def initcandidate():
        maxint = 0
        """
        to_do:
        """
        return maxint

    def candidate():
        """
        to_do:
        """"
        return True

    def getbounds(, vector):
        """
        to_do:
        return lower and upper bounds based on the formula provided in the paper
        """
        return lower_bound, upper_bound
