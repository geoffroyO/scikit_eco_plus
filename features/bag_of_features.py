from scikit_eco_plus.features.hog import hog_features
from scikit_eco_plus.models.kmeans import Kmeans

import numpy as np


class BagOfFeatures:
    def __init__(self, nclusters=256):
        self.n_cluster = nclusters
        self.hog = hog_features
        self.kmeans = None

    def fit(self, X):
        features = self.hog(X)
        features_tmp = features.reshape(features.shape[0] * features.shape[1], -1)
        self.kmeans = Kmeans(self.n_cluster)
        self.kmeans.fit(features_tmp)
        return features

    def predict(self, X, feature=False):
        if not feature:
            features = self.hog(X)
        else:
            features = np.copy(X)
        n, m, _ = features.shape
        features = features.reshape(n * m, -1)
        classification = self.kmeans.predict(features)
        classification = classification.reshape(n, m)

        bag = np.zeros((n, self.n_cluster))

        for idx, cluster in enumerate(classification):
            for c in cluster:
                bag[idx, c] += 1
        return bag
