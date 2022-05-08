#pca
import numpy as np
from scipy.linalg import eigh


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.tot_var = None
        self.alpha = None
        self.explained_variance = None

    def fit(self, X):
        N = len(X[0])
        X_centered = X - np.mean(X, axis=0)
        cov = np.cov(X_centered, rowvar=False)
        self.tot_var = np.sum(np.diag(cov))

        subset = [N - self.n_components, N - 1]
        W, V = eigh(cov, subset_by_index=subset)

        V_sorted = V[:, ::-1]
        self.alpha = V_sorted

    def transform(self, X):
        X_pca = X@self.alpha
        cov_pca = np.cov(X_pca, rowvar=False)
        self.explained_variance = np.array([var / self.tot_var for var in np.diag(cov_pca)])
        return X_pca

