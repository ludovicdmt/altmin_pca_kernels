import numpy as np
import copy
from scipy import optimize as opt


def _fit(X, n_comps, tolerance, cost_func_w, cost_func_c):
    n_observations = X.shape[0]
    n_features = X.shape[1]

    W = np.zeros((n_observations, n_comps))
    C = np.zeros((n_comps, n_features))

    Xfit = copy.copy(X)

    # Fit one component at a time
    for comp_idx in range(n_comps):

        # Initialize w and c
        w = np.random.randn(n_observations, 1)
        c = np.random.randn(1, n_features)

        iter = 1
        last_err = 0
        err = np.linalg.norm((w @ c) - Xfit, ord='fro')

        # while SC not satisfied, do
        while np.abs(err - last_err) > tolerance:
            last_err = err

            # Optimize w - one step
            res = opt.minimize(cost_func_w,  np.squeeze(w), args=(c, Xfit), method='CG', options={'maxiter': 1})
            w = np.reshape(res.x, (n_observations, 1))

            # Optimize c - one step
            res = opt.minimize(cost_func_c,  np.squeeze(c), args=(w, Xfit), method='CG', options={'maxiter': 1})
            c = np.reshape(res.x, (1, n_features))

            # update error and count
            pred_diff = w @ c - Xfit
            err = np.linalg.norm(pred_diff, ord='fro')
            iter = iter + 1

        print('PC{}: iterations={}, error={}'.format(comp_idx + 1, iter, err))
        Xfit = Xfit - (w @ c)
        W[:, comp_idx] = w[:, 0]
        C[comp_idx, :] = c[0, :]

    return W, C


def pca(X, n_comps, tolerance=0.001):
    """
    Fits PCA on the input data using the specified number of components.

    Parameters:
    X (array-like): Input data with shape (n_samples, n_features).
    n_comps (int): Number of components to fit.
    tolerance (float): Tolerance for convergence (default=0.001).

    Returns:
    W (array-like): Matrix of weights with shape (n_samples, total_n_comps).
    C (array-like): Matrix of coefficients with shape (total_n_comps, n_features).
    """

    def cost_func_w(predW, c, Xfit):
        predX = predW[:,np.newaxis] @ c
        pred_diff = predX - Xfit
        cost = np.power(np.linalg.norm(pred_diff, ord='fro'), 2)
        return cost

    def cost_func_c(predC, w, Xfit):
        predX = w @ predC[np.newaxis,:]
        pred_diff = predX - Xfit
        cost = np.power(np.linalg.norm(pred_diff, ord='fro'), 2)
        return cost

    return _fit(X, n_comps, tolerance, cost_func_w, cost_func_c)


def quadratically_regularized_pca(X, n_comps=20, alpha=1.0, tolerance=0.001):
    """
    Fits quadratically regularized PCA on the input data using the specified number
    of components.

    Parameters:
    X (array-like): Input data with shape (n_samples, n_features).
    n_comps (int): Number of components to fit.
    alpha (float): Regularization parameter. (default=1.0)
    tolerance (float): Tolerance for convergence. (default=0.001).

    Returns:
    W (array-like): Matrix of weights with shape (n_samples, total_n_comps).
    C (array-like): Matrix of coefficients with shape (total_n_comps, n_features).
    """

    def cost_func_w(predW, c, Xfit):
        predX = predW[:,np.newaxis] @ c
        pred_diff = predX - Xfit
        err = np.power(np.linalg.norm(pred_diff, ord='fro'), 2)
        l2_reg = alpha * np.sum(np.power(np.linalg.norm(predW), 2))# + alpha * np.sum(np.power(np.linalg.norm(c), 2))
        cost = err + l2_reg
        return cost

    def cost_func_c(predC, w, Xfit):
        predX = w @ predC[np.newaxis,:]
        pred_diff = predX - Xfit
        err = np.power(np.linalg.norm(pred_diff, ord='fro'), 2)
        l2_reg = alpha * np.sum(np.power(np.linalg.norm(predC), 2))# + alpha * np.sum(np.power(np.linalg.norm(w), 2))
        cost = err + l2_reg
        return cost

    return _fit(X, n_comps, tolerance, cost_func_w, cost_func_c)


def sparse_pca(X, n_comps=20, alpha=1.0, tolerance=0.001):
    """
    Fits sparse PCA on the input data using the specified number
    of components.

    Parameters:
    X (array-like): Input data with shape (n_samples, n_features).
    n_comps (int): Number of components to fit.
    alpha (float): Regularization parameter. (default=1.0)
    tolerance (float): Tolerance for convergence. (default=0.001).

    Returns:
    W (array-like): Matrix of weights with shape (n_samples, total_n_comps).
    C (array-like): Matrix of coefficients with shape (total_n_comps, n_features).
    """

    def cost_func_w(predW, c, Xfit):
        predX = predW[:,np.newaxis] @ c
        pred_diff = predX - Xfit
        err = np.power(np.linalg.norm(pred_diff, ord='fro'), 2)
        l2_reg = alpha * np.sum(np.linalg.norm(predW, ord=1))# + alpha * np.sum(np.linalg.norm(c, ord=1))
        cost = err + l2_reg
        return cost

    def cost_func_c(predC, w, Xfit):
        predX = w @ predC[np.newaxis,:]
        pred_diff = predX - Xfit
        err = np.power(np.linalg.norm(pred_diff, ord='fro'), 2)
        l2_reg = alpha * np.sum(np.linalg.norm(predC, ord=1))# + alpha * np.sum(np.linalg.norm(w, ord=1))
        cost = err + l2_reg
        return cost

    return _fit(X, n_comps, tolerance, cost_func_w, cost_func_c)


# all_bursts=np.load('all_bursts.npy')
# [W,C]=fit_pca(all_bursts, 20)

