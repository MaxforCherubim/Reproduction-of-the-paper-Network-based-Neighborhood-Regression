# For calling R functions
import os
os.environ['RPY2_CFFI_MODE'] = 'ABI'

import numpy as np
import scipy as sp

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

def generate_data(n=100, K=4, coef='full',
        sigma=1., seed=0):
    '''
    Generate data for the network community least squares problem.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    K : int
        The number of communities.
    coef : str
        The type of coefficient matrix. 'full' for full matrix, 'sgtn' for one non-zero coefficient, 'row' for one non-zero coefficient in each row.
    sigma : float
        The standard deviation of the noise.
    seed : int
        The random seed.

    Returns
    -------
    Z : numpy.ndarray
        The one-hot encoding of the cluster assignment.
    B : numpy.ndarray
        The probability matrix of the communities.
    P : numpy.ndarray
        The probability matrix of the graph.
    A : numpy.ndarray
        The adjacency matrix of the graph.    
    beta : numpy.ndarray
        The true coefficient vector.
    x : numpy.ndarray
        The predictor variable.
    y : numpy.ndarray
        The response variable.
    '''
    np.random.seed(seed)

    Z = np.eye(K)[np.random.choice(K, n)]
    B_triu = np.triu(np.random.uniform(0, 0.5, [K,K]), k=0)
    B = (B_triu + B_triu.T)/2 + np.eye(K) * 0.5
    P = Z @ B @ Z.T    

    A_triu = np.triu(np.random.binomial(1, P), k=1)        
    A = A_triu + A_triu.T
    np.fill_diagonal(A, 1)
    A = A.astype(np.float32)

    if coef=='full':
        beta = np.random.randn(K,K)
    elif coef=='sgtn':
        beta = np.random.randn(1) * np.ones((K,K))
    elif coef=='row':
        beta = np.ones((K,1)) @ np.random.randn(1,K)
    x = np.random.randn(n,1)
    y = ((Z @ beta @ Z.T) * A) @ x + sigma * np.random.randn(n,1)

    return Z, B, P, A, beta, x, y



def community_detection(A, K=4):
    '''
    Perform community detection on the graph represented by the adjacency matrix A.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix of the graph.
    K : int
        The number of communities to detect.

    Returns
    -------
    C : numpy.ndarray
        The cluster assignment of each node.
    Z : numpy.ndarray
        The one-hot encoding of the cluster assignment.
    '''
    U, s, VT = np.linalg.svd(A)
    W = U[:, :K]

    # W = A @ U
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    W = np.nan_to_num(W, 0.)

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0).fit(W)
    C = kmeans.predict(W)

    Z = np.eye(K)[C]

    return C, Z



def _pinv_extended(x, rcond=1e-3):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.
    Code adapted from numpy.
    """
    x = np.asarray(x)
    x = x.conjugate()
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis],
                                               np.transpose(u)))
    return res, s_orig


def _cov_hc3(h, pinv_wexog, resid):
    het_scale = (resid/(1-h))**2

    # sandwich with pinv(x) * diag(scale) * pinv(x).T
    # where pinv(x) = (X'X)^(-1) X and scale is (nobs,)
    cov_hc3_ = np.dot(pinv_wexog, het_scale[:,None]*pinv_wexog.T)
    return cov_hc3_



def solve(X, y):
    pinv_wexog, singular_values = _pinv_extended(X)
    normalized_cov = np.dot(pinv_wexog, np.transpose(pinv_wexog))
    h = np.diag(np.dot(X, np.dot(normalized_cov, X.T)))
    hbeta = np.dot(pinv_wexog, y)
    resid = y - X @ hbeta
    cov = _cov_hc3(h, pinv_wexog, resid)
    std_dev = np.sqrt(np.diag(cov))

    return hbeta, std_dev


def CLSE(y, x, A, Z, coef='full', return_std=False):
    '''
    Comminity-wise least squares estimator.

    Parameters
    ----------
    y : numpy.ndarray
        The response variable.
    x : numpy.ndarray
        The predictor variable.
    A : numpy.ndarray
        The adjacency matrix of the graph.
    Z : numpy.ndarray
        The one-hot encoding of the cluster assignment.
    ceof : numpy.ndarray
        The type of coefficient matrix, can be 'full', 'sgtn', 'row'.

    Returns
    -------
    hbeta : numpy.ndarray
        The estimated coefficient estimates.
    '''
    if len(y.shape) == 1:
        y = y[:,None]
    if len(x.shape) == 1:
        x = x[:,None]
    K = Z.shape[1]

    if coef=='full':
        Mk = [np.diag(Z[:,k]) @ (x * A).T @ Z for k in range(K)]
        hbeta, std = zip(*[solve(Mk[k][Z[:,k]==1], y[:,0][Z[:,k]==1]) for k in range(K)])
        hbeta = np.array(hbeta)
        std = np.array(std)

    elif coef=='sgtn':
        beta0 = (x.T @ A @ y) / (x.T @ np.linalg.matrix_power(A, 2) @ x + 1e-16)
        hbeta = np.ones((K,K)) * beta0

    elif coef=='row':
        beta0 = solve(A @ np.diag(x[:,0]) @ Z, y[:,0])[0]
        hbeta = np.ones((K,1)) @ beta0[None,:]

    if not return_std:
        return hbeta
    else:
        return hbeta, std


def LR(y, x, A, Z):
    return CLSE(y, x, np.eye(A.shape[0]), Z)

def LR1(y, x, A, Z):
    return CLSE(y, x, np.ones_like(A), Z)    



from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


def permute_columns(Z_hat, Z):
    '''
    Permute the columns of Z_hat to minimize the Hamming distance to Z.

    Parameters
    ----------
    Z_hat : numpy.ndarray
        The predicted cluster assignment.
    Z : numpy.ndarray
        The true cluster assignment.

    Returns
    -------
    Z_hat_permuted : numpy.ndarray
        The permuted predicted cluster assignment.
    '''
    # Calculate pairwise distance matrix
    dist_matrix = pairwise_distances(Z.T, Z_hat.T, metric='hamming')

    # Use the Hungarian algorithm to find the optimal column permutation
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Permute the columns of Z_hat
    Z_hat_permuted = Z_hat[:, col_ind]

    return Z_hat_permuted

from pathlib import Path
os.environ['R_HOME'] = str(Path.cwd() / '.pixi/envs/default/lib/R')
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

# Create a converter that starts with rpy2's default converter
# to which the numpy conversion rules are added.
np_cv_rules = default_converter + numpy2ri.converter

import rpy2.robjects as robjects
# install.packages('netcoh', repos = NULL, type="source")

robjects.r('''
suppressMessages(library(Matrix))
library("netcoh")
run_netcoh <- function(y, x, A){
    lambdaseq <- 10^seq(-3, 1, length.out = 100)

    Y <- matrix(y, ncol=1)
    X <- matrix(x, ncol=1)
    A <- as.matrix(A)

    res <- rncregpath(A,lambdaseq,Y,X, cv=5)
    model.cv <- res$models[[res$cv.1sd.index]]

    Y_hat <- matrix(model.cv$alpha, ncol=1) + X %*% model.cv$beta

    lam <- lambdaseq[res$cv.1sd.index]

    return(list('yhat'=Y_hat, 'alpha'=model.cv$alpha, 'beta'=model.cv$beta, 'lam'=lam))
}
''')

run_netcoh = robjects.globalenv['run_netcoh']
