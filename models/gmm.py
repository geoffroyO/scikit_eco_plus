# GMM implementation in pytorch

import torch
from tqdm import tqdm

from scikit_eco_plus.models.kmeans import Kmeans
import numpy as np


class GMM:
    def __init__(self, n_clusters, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        self.data = None
        self.means = None
        self.covs = None
        self.weights = None

    def _kmeans_init(self):
        _, d = self.data.shape
        state = self.random_state
        while True:
            model = Kmeans(self.n_clusters, random_state=state)
            model.fit(self.data)
            tmp = [self.data[model.labels == k].size for k in range(self.n_clusters)]
            if 0 in tmp:
                state = np.random.randint(0, 4000)
            else:
                break
        means = np.zeros((self.n_clusters, d))
        covs = np.zeros((self.n_clusters, d, d))
        weights = np.zeros(self.n_clusters)

        for k in range(self.n_clusters):
            X_k = self.data[model.labels == k]
            if X_k.size == 0:
                means[k] = np.zeros(d)
                weights[k] = 0
            else:
                means[k] = np.mean(X_k, axis=0)
                weights[k] = len(X_k) / len(self.data)

            covs[k] = np.eye(d)
        return torch.from_numpy(means), torch.from_numpy(covs), torch.from_numpy(weights)

    def fit(self, data):
        self.data = np.copy(data)
        means, covs, weights = self._kmeans_init()
        data = torch.from_numpy(self.data)
        N, d = self.data.shape

        for _ in range(self.max_iter):
            CholCov = torch.linalg.cholesky(covs)
            probs = torch.distributions.MultivariateNormal(means, scale_tril=CholCov)

            # E step
            gamma = torch.reshape(weights, (1, self.n_clusters)) * torch.exp(
                probs.log_prob(torch.reshape(data, (N, 1, d))))
            gamma /= torch.sum(gamma, dim=1, keepdim=True)
            N_k = torch.sum(gamma, dim=0)

            # M step
            weights = N_k / N

            means = torch.sum(torch.reshape(gamma, (N, self.n_clusters, 1)) * torch.reshape(data, (N, 1, d)), dim=0)
            means /= torch.reshape(N_k, (self.n_clusters, 1))

            centered = torch.reshape(data, (1, N, d)) - torch.reshape(means, (self.n_clusters, 1, d))
            centered_gamma = centered * torch.reshape(torch.t(gamma), (self.n_clusters, N, 1))

            sample_covs = torch.matmul(torch.transpose(centered_gamma, 1, 2), centered)
            covs = sample_covs / torch.reshape(N_k, (self.n_clusters, 1, 1))
            covs = covs + 1e-8 * torch.eye(d).repeat(self.n_clusters, 1, 1)

        self.means = means.numpy()
        self.covs = covs.numpy()
        self.weights = weights.numpy()

    def predict_proba(self, data):
        N, d = data.shape

        means, covs, weights = torch.tensor(self.means), torch.tensor(self.covs), torch.tensor(self.weights)
        CholCov = torch.linalg.cholesky(covs)
        probs = torch.distributions.MultivariateNormal(means, scale_tril=CholCov)

        gamma = torch.reshape(weights, (1, self.n_clusters)) * torch.exp(
            probs.log_prob(torch.reshape(torch.tensor(data), (N, 1, d))))
        gamma /= torch.sum(gamma, dim=1, keepdim=True)

        return gamma.numpy()
