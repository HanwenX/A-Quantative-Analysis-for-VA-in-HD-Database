import numpy as np 

# returns true if bb1 contains bb2, false otherwise
# each bbox is data_dim by 2
def bb_contain(bb1, bb2):
    bb1, bb2 = np.array(bb1), np.array(bb2)
    # check lower left
    if np.any(bb1[0,:] > bb2[0,:]):
        return False
    # check upper right
    if np.any(bb1[1,:] < bb2[1,:]):
        return False
    return True

# computes and returns the area (volume) of a bbox
def bb_area(bb):
    lengths = bb[1,:] - bb[0,:]
    return np.prod(lengths)

# computes and returns the minimal bouning box containing all of the given bbs
def bb_list_merge(bb_list):
    lower_left = np.min(np.vstack([bb[0,:] for bb in bb_list]), axis=0)
    upper_right = np.max(np.vstack([bb[1,:] for bb in bb_list]), axis=0)
    return np.vstack([lower_left,upper_right])

