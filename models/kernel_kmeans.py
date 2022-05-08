#Spectral clustering
from scipy.linalg import eigh
import numpy as np
from scikit_eco_plus.models.kmeans import Kmeans


class KernelKmeans:
    def __init__(self, n_clusters, kernel, max_iter=300, tol=10e-3, random_state=42, **kwargs):
        self.kernel = kernel
        self.n_cluster = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.labels = None
        self.data = None
        self.K_data = None

    def fit(self, X):
        N = len(X)
        K = self.kernel(X, X)

        subset = [N - self.n_cluster, N - 1]
        _, V = eigh(K, subset_by_index=subset)

        Z = V / np.linalg.norm(V, axis=1, ord=2)[:, np.newaxis]

        model = Kmeans(self.n_cluster, self.max_iter, self.tol, self.random_state)
        model.fit(Z)

        self.labels = model.labels
        self.data = X
        self.K_data = K

    def fit_affinity(self, K):
        N = len(K)

        subset = [N - self.n_cluster, N - 1]
        _, V = eigh(K, subset_by_index=subset)

        Z = V / np.linalg.norm(V, axis=1, ord=2)[:, np.newaxis]

        model = Kmeans(self.n_cluster, self.max_iter, self.tol, self.random_state)
        model.fit(Z)

        return model.labels

    def predict(self, X):
        K_X = self.kernel(X, X)
        K_data_X = self.kernel(self.data, X)
        dist = np.zeros((2, len(X)))
        for s in range(2):
            indx = self.labels == s
            card = np.sum(indx)
            th1 = np.diag(K_X)
            th2 = -2 * np.sum(K_data_X[indx, :], axis=0) / card
            th3 = np.sum(self.K_data[indx, indx]) / card ** 2
            dist[s] = th1 + th2 + th3
        return [np.argmin(i) for i in dist.T]
