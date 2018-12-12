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



def get_means(data, k):
    mean = np.mean(data, axis=0)
    dists = np.sqrt(np.sum(np.power(data-mean,2), axis=1))
    furthest = data[np.argmax(dists),:]
    means = [furthest]
    for i in range(k-1):
        mean = np.mean(means, axis=0)
        dists = np.sqrt(np.sum(np.power(data-mean,2), axis=1))
        furthest = data[np.argmax(dists),:]
        means.append(furthest)
    return np.array(means)


def cluster_data(num_points, dim, k, var=0.01):
    np.random.seed(1)

    candidate_means = np.random.rand(1000000,45)[:,:dim]
    means = get_means(candidate_means, k)

    noise_cov = np.eye(dim) * var
    noise_mean = np.zeros(dim)
    

    noise = np.random.multivariate_normal(noise_mean, noise_cov, num_points)

    num_per_cluster = num_points // k
    data = np.vstack([np.repeat(means[i][np.newaxis,:], num_per_cluster, axis=0) for i in range(k)])
    data = data + noise
    data = np.clip(data, a_min=0, a_max=1)
    np.random.shuffle(data)
    return data
