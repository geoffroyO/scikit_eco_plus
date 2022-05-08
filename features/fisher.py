# Reference Jorge Sanchez, Florent Perronnin, Thomas Mensink, Jakob Verbeek https://hal.inria.fr/hal-00830491v2/document
import numpy as np
from scikit_eco_plus.models.gmm import GMM

from tqdm import tqdm


class Fisher:
    def __init__(self, X, K):
        self.K = K
        self.N, self.T, self.D = X.shape
        self.X = X

    def fit(self):
        fv = np.zeros((self.N, (2 * self.D + 1) * self.K))
        for idx, img in tqdm(enumerate(self.X)):
            gmm = GMM(self.K)
            gmm.fit(img)
            means, cov, weights = gmm.means, gmm.covs, gmm.weights
            gamma = gmm.predict_proba(img).T
            cov_diag = np.array([np.diag(c) for c in cov])

            s0 = np.sum(gamma, axis=0)
            s1 = gamma @ img
            s2 = gamma @ (img ** 2)

            grad_alpha = np.zeros(self.K)
            for k in range(self.K):
                grad_alpha[k] = (s0[k] - self.T * weights[k]) / np.sqrt(weights[k] + 10e-5)

            grad_means = np.zeros((self.K, self.D))
            for k in range(self.K):
                grad_means[k] = s1[k] - s0[k] * means[k] / np.sqrt(weights[k] * cov_diag[k] + 10e-5)

            grad_cov = np.zeros((self.K, self.D))
            for k in range(self.K):
                grad_cov[k] = (s2[k] ** 2 - 2 * means[k] * s1[k] + (means[k] ** 2 - cov_diag[k]) * s0[k]) / np.sqrt(
                    2 * weights[k] * cov_diag[k] + 10e-5)

            fv_tmp = np.concatenate([grad_alpha, grad_means.reshape(self.K * self.D), grad_cov.reshape(self.K * self.D)])
            fv_tmp = np.sign(fv_tmp) * np.sqrt(np.abs(fv_tmp))
            fv[idx] = fv_tmp / (np.linalg.norm(fv_tmp, ord=2) + 10e-5)

        return fv
