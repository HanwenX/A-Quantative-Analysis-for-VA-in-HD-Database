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

# checks if point is contained in boundingbox
def point_contain(bb, point):
    return bb_contain(bb, [point,point])

# computes and returns the area (volume) of a bbox
def bb_area(bb):
    lengths = np.abs(bb[1,:] - bb[0,:])
    return np.prod(lengths)

def bb_margin(bb):
    lengths = bb[1,:] - bb[0,:]
    return np.sum(lengths)

# returns the area of overlap between 2 bounding boxes
def bb_overlap(bb1, bb2):
    min_ur = np.min(np.vstack([bb1[1], bb2[1]]), axis=0)
    max_ll = np.max(np.vstack([bb1[0], bb2[0]]), axis=0)
    return bb_area(np.vstack([max_ll, min_ur]))


# computes and returns the minimal bouning box containing all of the given bbs
def bb_list_merge(bb_list):
    lower_left = np.min(np.vstack([bb[0,:] for bb in bb_list]), axis=0)
    upper_right = np.max(np.vstack([bb[1,:] for bb in bb_list]), axis=0)
    return np.vstack([lower_left,upper_right])

# computes minimum distanc from a query point to a bounding box
def min_dist(bb, point):
    if point_contain(bb, point):
        return 0.0
    widths = bb[1,:] - bb[0,:]
    centers = (bb[1,:] + bb[0,:])/2
    ds = np.maximum(np.abs(centers-point) - widths/2,0)
    return np.sqrt(np.sum(np.power(ds,2)))

def max_dist(bb, point):
    widths = bb[1,:] - bb[0,:]
    centers = (bb[1,:] + bb[0,:])/2
    ds = np.maximum(np.abs(centers-point) + widths/2,0)
    return np.sqrt(np.sum(np.power(ds,2)))


# accepts Point object and query point, computes distance
def dist(Point, q):
    return np.sqrt(np.sum(np.power(q-Point.mbb[0,:],2)))









