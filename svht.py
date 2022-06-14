# use results of Gavish and Donoho Paper
# used the following code from http://www.pyrunner.com/weblog/2016/08/01/optimal-svht/
# to calculate the SVHT... pretty cool stuff, fairly inscrutable

import numpy as np
from scipy.linalg import svd, svdvals

def omega_approx(beta):
    """Return an approximate omega value for given beta. Equation (5) from Gavish 2014."""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def lambda_star(beta):
    """Return lambda star for given beta. Equation (11) from Gavish 2014."""
    return np.sqrt(2 * (beta + 1) + (8 * beta) /
                   (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1)))


def svht(X, sigma=None, sv=None):
    """Return the optimal singular value hard threshold (SVHT) value.
    `X` is any m-by-n matrix. `sigma` is the standard deviation of the
    noise, if known. Optionally supply the vector of singular values `sv`
    for the matrix (only necessary when `sigma` is unknown). If `sigma`
    is unknown and `sv` is not supplied, then the method automatically
    computes the singular values."""

    try:
        m, n = sorted(X.shape)  # ensures m <= n
    except:
        raise ValueError('invalid input matrix')
    beta = m / n  # ratio between 0 and 1
    if sigma is None:  # sigma unknown
        if sv is None:
            sv = svdvals(X)
        sv = np.squeeze(sv)
        if sv.ndim != 1:
            raise ValueError('vector of singular values must be 1-dimensional')
        return np.median(sv) * omega_approx(beta)
    else:  # sigma known
        return lambda_star(beta) * np.sqrt(n) * sigma