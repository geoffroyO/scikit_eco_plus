import numpy as np
from scipy.linalg import eigh
from scikit_eco_plus.models.models import Model


class KernelPCA(Model):
    def __init__(self, kernel='linear', **kwargs):
        super().__init__(kernel, **kwargs)

        self.alpha = None
        self.center = None
        self.X = None

    def fit(self, X):
        self.X = X
        K, self.center = self.kernel(X, X, centered=True)
        W, V = eigh(K)
        W, V = W[::-1], V[:, ::-1]
        support = W > 0
        W, V = W[support], V[:, support]

        self.alpha = np.column_stack([V[:, i] / np.sqrt(W[i]) for i in range(len(W))])

    def fit_transform(self, X, n_components=None):
        self.fit(X)
        self.transform(X, n_components)

    def transform(self, X, components=None):
        if components is None:
            components = self.alpha.shape[1]
        K = self.kernel(X, self.X)
        K = K - np.mean(K, axis=1)[:, None] - self.center
        return K@self.alpha[:, :components]

