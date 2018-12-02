from skimage.color import rgb2gray
from utils import unpickle, extract_lbp_feature
import numpy as np

# Unpickle image data and convert to gray scale images.
a = unpickle('data/cifar-100-python/train')
data = np.concatenate((a['data'][:, :1024].reshape(50000, 32, 32, 1), \
                      a['data'][:, 1024:2048].reshape(50000, 32, 32, 1), \
                      a['data'][:, 2048:3072].reshape(50000, 32, 32, 1)), axis=-1)
data = rgb2gray(data)

# Generate lbp feature vectors of dimension n where n <= 256.
# For our case, at most 45 dimension is needed.
for n in xrange(2, 46):
    print 'in dim', n
    features = extract_lbp_feature(data, n)
    np.savetxt('data/fv/fv'+str(n)+'d.txt', features)
