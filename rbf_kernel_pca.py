from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """ implemention of RBF kernel PCA

    paraters
    ----------
    X: {Numpy ndarray}, shape = [n_samples, n_features]

    gamma: float
        a tooning parameter of RBF kernel
    
    n_components: int
        the number of returning PCA

    return
    ----------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
        projected dataset

    """

    # calculate each data's distance of M X N dimension dataset
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)

    K = exp(-gamma * mat_sq_dists)

    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    X_pc = np.column_stack((eigvecs[:, i]
                            for i in range(n_components)))

    alphas = np.column_stack((eigvecs[:, i]
                              for i in range(n_components)))

    lambdas = [eigvals[i] for i in range(n_components)]
    return alphas, lambdas
