#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supervised dictionnary learning with orthogonal components and regularization.

Author: Ludovic Darmet
email: ludovic.darmet@gmail.com
"""
__author__ = "Ludovic Darmet"

import numpy as np
from scipy import optimize as opt
from pyemd import emd_samples

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class OrthoDictLearning(BaseEstimator, TransformerMixin):
    """Dictionnary learning with orthogonals components.
    
    Fit a regularized and supervised dictionnary learning model.
    The input data are decomposed into a matrix of components and a matrix of weights that minimize the reconstruction error.
    The components are constrained to be orthogonal and the weights are constrained to be sparse. The weights can also be
    constrained to be discriminative of the class (latent space).
    
   Parameters
    ----------
    n_comps : int
        Number of components to fit.
    alpha1 : float, optional
        Regularization parameter for sparsity. Default is 10.0.
    alpha2 : float, optional
        Regularization parameter for discriminability. Default is 10.0.
    reg : str, optional
        Regularization type ('l1' or 'l2'). Default is 'l1'.
    tolerance : float, optional
        Tolerance for convergence. Default is 0.001.
    verbose : int, optional

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        dictionary atoms extracted from the data
    weights_ : ndarray of shape (n_samples, n_components)
        weights for each sample
    error_ : array
        vector of errors at each iteration
    n_comps : int
        Number of features seen during :term:`fit`.
    """

    def __init__(self, n_comps, alpha1 = 10.0, alpha2= 10.0, reg='l1', tolerance=0.001, verbose=0):
        """Initialize the model."""
        assert reg in ['l1', 'l2', 'None'], "Regularization type must be 'l1' or 'l2'."
        if (alpha2 != 0):
            assert y is not None, "Discriminative regularization requires class labels."

        self.n_comps = n_comps
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.reg = reg
        self.tolerance = tolerance
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the model to the data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,), default=None
            The class labels. Mandatory if alpha2 is not 0.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(X, y)
        return self
    
    def fit_transform(self, X, y=None):
        """Fit the model to the data and return the transformed data.
        The components and the weight matrix are learnt alternatively using conjugate gradient descent.
        A proximal operator is used to enforce orthogonality of the components. Sparsity and discriminability
        of the weights are enforced using a regularization term in the cost function.
        """
        if self.alpha2 != 0:
            assert y is not None, "Discriminative regularization requires class labels."
        self.error_ = []
        n_observations = X.shape[0]
        n_features = X.shape[1]

        W = np.zeros((n_observations, self.n_comps))
        C = np.zeros((self.n_comps, n_features))

        Xfit = np.copy(X)

        # Fit one component at a time
        for comp_idx in range(self.n_comps):

            # Initialize weights and components randomly
            # TODO: Initialize with SVD
            w = np.random.randn(n_observations, 1)
            c = np.random.randn(1, n_features)

            iter = 1
            last_err = 0
            err = np.linalg.norm(np.dot(w, c) - Xfit, ord='fro')

            # while the reconstruction error (not the optimized loss) is progressing, do
            while np.abs(err - last_err) > self.tolerance:
                last_err = err

                # Optimize the weight matrix - one step
                res = opt.minimize(self._cost_func_w, np.squeeze(w), args=(c, Xfit, y), method='CG', options={'maxiter': 1})
                w = np.reshape(res.x, (n_observations, 1))

                # proximal projection to ensure positivity of the weights
                # w = np.maximum(w, 0)

                # Optimize the components - one step
                res = opt.minimize(self._cost_func_c, np.squeeze(c), args=(w, Xfit), method='CG', options={'maxiter': 1})
                c = np.reshape(res.x, (1, n_features))
                # normalization
                c /= np.linalg.norm(c)
                # proximal projection to ensure orthogonality of the basis
                if comp_idx > 0:
                    c = self._orthogonal_projection(c, C[:comp_idx, :])

                # update error of reconstruction (different from the loss function!)
                pred_diff = np.dot(w, c) - Xfit
                err = np.linalg.norm(pred_diff, ord='fro')
                iter = iter + 1

            if self.verbose > 0:
                print('PC{}: iterations={}, error={}'.format(comp_idx + 1, iter, err))

            self.error_.append(err)
            # Compute residual to fit the next component
            Xfit = Xfit - np.dot(w, c)
            W[:, comp_idx] = w[:, 0]
            C[comp_idx, :] = c[0, :]
        # Save the components and weights
        self.components_ = C
        self.weights_ = W

        return W
    
    def transform(self, X):
        """Transform the input data using the fitted components."""
        check_is_fitted(self)
        return np.dot(X, self.components_.T)

    def _projection(self, w, v):
        "Function to compute the projection of vector w onto vector v."
        return np.dot(w, v) / np.dot(v, v) * v

    def _orthogonal_projection(self, w, w_base):
        "Function to make vector w orthogonal to the basis w_base."
        # Normalize the basis vectors
        w_base_norm = [v / np.linalg.norm(v) for v in w_base]
        
        # Compute the projection of w onto each basis vector and subtract it from w
        for v in w_base_norm:
            w -= self._projection(w, v)
        return w / np.linalg.norm(w)
    
    def _cost_func_w(self, predW, c, Xfit, y):
        """Cost function to optimize the weight matrix.
        Some regularization terms are added to the cost function to enforce sparsity and discriminability.
        """
        predX = np.dot(predW[:,np.newaxis], c)
        pred_diff = predX - Xfit
        cost = np.linalg.norm(pred_diff, ord='fro')**2

        # Regularization term for a sparse activation matrix
        if self.reg == 'l2':
            reg_activ = self.alpha1 * np.sum(np.linalg.norm(predW)**2)
        elif self.reg == "l1":
            reg_activ = self.alpha1 * np.sum(np.linalg.norm(predW, ord=1))
        else:
            reg_activ = 0

        # Regularization term for discriminative PCA
        if y is not None:
            classes = np.unique(y)
            reg_dis = 0
            for class_ in classes:
                class_subsample = predW[y == class_]
                rest_subsample = predW[y != class_]
                # Wasserstein distance for the class at hand vs the rest
                reg_dis += self.alpha2/len(classes) * emd_samples(class_subsample, rest_subsample)
        else:
            reg_dis = 0
        # print(f"Cost: {cost}, reg_activ: {reg_activ}, reg_dis: {reg_dis}")
        cost = cost + reg_activ + reg_dis
        return cost

    def _cost_func_c(self, predC, w, Xfit):
        """Cost function to optimize the components matrix.
        Orthogonality is enforced by the proximal operator after the optimization step.
        """
        predX = np.dot(w, predC[np.newaxis,:])
        pred_diff = predX - Xfit
        cost = np.linalg.norm(pred_diff, ord='fro')**2
        return cost

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Generate random data
    X = np.random.normal(size=(100, 10))
    X, y = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    X=StandardScaler().fit_transform(X)

    # Run the dictionary learning
    model = OrthoDictLearning(n_comps=3, alpha1=5, alpha2=20, reg='l1')
    model.fit(X, y)
    X_proj = model.transform(X)
    W_opt = model.weights_
    C_opt = model.components_
    print(W_opt.shape, C_opt.shape)
    product = np.dot(C_opt.T, C_opt)  
    plt.scatter(W_opt[:,0], W_opt[:,1], c=y)
    plt.show()

    # Check if the product is an identity matrix
    is_orthonormal = np.allclose(product, np.eye(product.shape[0]))

    print(is_orthonormal)
    print(product - np.eye(product.shape[0]))