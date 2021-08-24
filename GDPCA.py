import numpy as np
import scipy.sparse as sp
import timeit
import math
from scipy.sparse.linalg import gmres
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def matrix_preprocessing(A, delta, sigma, S):
    ##############################################################################
    ### matrix_preprocessing - This method provides Laplacian Regularisation of matrix A
    ###                        with reweighting by covariance matrix S.
    ### Input:
    ### A - a symmetrix (adjacency/similarity) matrix (n x n), where n is the number of nodes (sp.spmatrix);
    ### delta - a reweighting parameter (float in (0,1));
    ### sigma - the power for Laplacian Regularization  (float in [0, 0.5, 1]);
    ### S - a covariance matrix (n x n), (np.ndarray).
    ### Output:
    ### A_hat - is a result of Laplacian Regularization of matrix A (np.ndarray);
    ### S_ - is a reweighted part of covariance matrix (n x n), (np.ndarray).
    ##############################################################################
    nnodes = A.shape[0]
    A = A + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    lsigma = sigma - 1
    rsigma = - sigma
    wsigma = -2 * sigma + 1

    D_l = sp.diags(np.power(D_vec, lsigma))
    D_r = sp.diags(np.power(D_vec, rsigma))
    Dw = sp.diags(np.power(D_vec, wsigma))
    S_ = S @ Dw
    A_hat = D_l @ A @ D_r + delta * S_
    return S_, A_hat

def PI(A, Z, Y, iter_, alpha):
    ##############################################################################
    ### PI - Power Iteration.
    ### This is an iterative method for the numerical solution of the linear system.
    ### Input:
    ### A - a symmetrix (adjacency/similarity) matrix (n x n), where n is the number of nodes (sp.spmatrix);
    ### Z - an initialization of solution of the linear system (np.ndarray);
    ### Y - an availabel label matrix (n x k) (sp.spmatrix, np.ndarray);
    ### iter_ - the number of iterations (int);
    ### alpha - a jump factor of Personalized pagerank (float in (0, 1)).
    ### Output:
    ### Z - solution of linear system (np.ndarray);
    ### time_ - computation time (float)
    ##############################################################################
    A = np.copy(A)
    Z = np.copy(Z)
    Y = np.copy(Y)
    time_ = 0
    for _ in range(iter_):
        start = timeit.default_timer()
        Z = alpha * A @ Z  + (1 - alpha) * Y
        Z = normalize(Z, 'l1')
        time_ += (timeit.default_timer() - start)
    return Z, time_

def GMRES(A, Y, alpha, k, tol):
    ##############################################################################
    ### GMRES - Generalized Minimal Residual Method.
    ### https://epubs.siam.org/doi/abs/10.1137/0907058
    ### This is an iterative method for the numerical solution of the linear system.
    ### Input:
    ### A - a symmetrix (adjacency/similarity) matrix (n x n), where n is the number of nodes (sp.spmatrix);
    ### k - the number of classes (int);
    ### Y - an availabel label matrix (n x k) (sp.spmatrix, np.ndarray);
    ### tol - tolerace is the real value (float).
    ### Output:
    ### Z - solution of linear system (np.ndarray);
    ### time_ - computation time (float)
    ##############################################################################
    time_ = 0
    A = np.copy(A)
    Y = np.copy(Y)
    Z = []
    for j in range(k):
        start = timeit.default_timer()
        temp_ = gmres(A, (1 - alpha) * Y[:, j], tol=tol)[0]
        time_ += (timeit.default_timer() - start)
        Z.append([temp_])
    return np.concatenate(Z).T, time_

def GDPCA(X, A, Y, delta=1, sigma=1, alpha=0.9,
          svd_error=1e-03, iter_=10, tol=1e-03, MMx=np.NAN):
    svd = TruncatedSVD(n_components=1, algorithm='randomized')
    nnodes = A.shape[0]

    if np.isnan(MMx).all():
        Xn = np.copy(X)
        Xn = Xn.T
        Xn = Xn - np.median(Xn, axis=0)
        MMx = np.dot(Xn.T, Xn) / (Xn.shape[0] - 1)
    mmc, AHAT = matrix_preprocessing(A, delta, sigma, MMx)
    rex = (np.identity(nnodes) - alpha * AHAT)
    svd.fit(mmc)
    gamma = svd.singular_values_[0]
    if (1 + delta * (gamma + svd_error)) <= 1 / alpha:
        print('PI')
        Z, tm = PI(A=AHAT, Z=Y, Y=Y, iter_=iter_, alpha=alpha)
    else:
        print('GMRES')
        Z, tm = GMRES(A=rex, Y=Y, alpha=alpha, k=Y.shape[1], tol=tol)
    return Z, tm

