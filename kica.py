
import numpy as np
from scipy.linalg import sqrtm, svd

def kica(xx):

    # Kurtosis Maximization ICA.

    # Data whitening
    xxm = xx - xx.mean(axis=1).reshape(-1, 1)
    a = sqrtm(np.linalg.inv(np.cov(xx))).real
    yy = np.dot(a, xxm)
    
    # Kurtosis Maximization ICA itself.
    ss = (yy**2).sum(axis=0)
    b = ss * yy
    c = np.dot(b, yy.T)
    [W, ss, vvt] = svd(c)

    return W, yy
