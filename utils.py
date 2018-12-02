import numpy as np
from scipy.ndimage import convolve

# This is the python2 version of unpickling file
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# This is the python3 version of unpicking file
#def unpickle(file):
#    import pickle
#    with open(file, 'rb') as fo:
#        dict = pickle.load(fo, encoding='bytes')
#    return dict

def extract_lbp_feature(data, n):
    """
    This function takes data of shape N x H x W and return
    extracted local binary pattern of shape N x n where n
    is a specified dimension for the final feature vector.

    n <= 256 is required
    """
    assert n <= 256
    N, H, W = data.shape
    patchSize = 3
    features = np.zeros((N, n))
    lbp = np.zeros((N, H, W), dtype=int)
    mask = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]]).reshape(1, 3, 3)
    channelPower = 0
    for i in xrange(patchSize):
        for j in xrange(patchSize):
            if i == patchSize//2 and j == patchSize//2:
                continue
            tmpMask = mask.copy()
            tmpMask[0, i, j] = 1
            channel = convolve(data, tmpMask, mode='constant')
            channel[channel > 0] = 1
            channel[channel <= 0] = 0
            lbp += channel.astype(int) * 2**channelPower
            channelPower += 1
    for i in xrange(N):
        features[i, :], k = np.histogram(lbp[i, :, :].ravel(), bins=n)

    # Normalize features s.t. all feature values are in range [0, 1]
    features = features/np.sqrt(np.sum(features**2, axis=1, keepdims=True))
    return features


