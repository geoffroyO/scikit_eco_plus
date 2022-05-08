#kmeans
import numpy as np
from scipy.spatial.distance import cdist

from tqdm import tqdm


class Kmeans:
    def __init__(self, n_clusters, max_iter=300, tol=10e-4, random_state=42):
        self.centroids = None
        self.labels = None
        self.n_cluster = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        idx_init = np.random.choice(len(X), self.n_cluster, replace=False)

        centroids = X[idx_init]
        dist_centroids = cdist(X, centroids, metric='euclidean')

        points_assignments = np.array([np.argmin(d) for d in dist_centroids])

        for _ in range(self.max_iter):
            centroids = np.zeros((self.n_cluster, len(X[0])))
            for k, cluster in enumerate(range(self.n_cluster)):
                centroids[k] = np.mean(X[points_assignments == cluster], axis=0)

            dist_centroids = cdist(X, centroids, metric='euclidean')

            if np.max(dist_centroids) <= self.tol:
                break

            points_assignments = np.array([np.argmin(d) for d in dist_centroids])

        self.centroids = centroids
        self.labels = points_assignments

    def predict(self, X):
        dist_centroids = cdist(X, self.centroids, metric='euclidean')
        return np.array([np.argmin(d) for d in dist_centroids])
