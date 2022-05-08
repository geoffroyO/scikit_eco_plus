import numpy as np
from scipy.spatial.distance import cdist


def center_kernel(K):
    n = len(K)
    one_n = np.ones((n, n)) / n
    center = np.mean(K, axis=1) - np.mean(K)
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n, center


class PolynomialKernel:
    def __init__(self, degree=3, gamma=1, coef0=1):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def __str__(self):
        return f"Noyau polynomial de degré {self.degree} de pente {self.gamma} et de coefficient nul {self.coef0}."

    def kernel(self, X, Y, centered=False):
        K = X @ Y.T
        K *= self.gamma
        K += self.coef0
        K **= self.degree
        if centered:
            K = center_kernel(K)
        return K


class GaussianKernel:
    def __init__(self, l=1):
        self.l = l

    def __str__(self):
        return f"Noyau gaussien de paramètre {self.l}."

    def kernel(self, X, Y, centered=False):
        K = cdist(X, Y, metric="sqeuclidean")
        K = np.exp(-self.l * K)
        if centered:
            K = center_kernel(K)
        return K


class LaplacianKernel:
    def __init__(self, l=1):
        self.l = l

    def __str__(self):
        return f'Noyau laplacien de paramètre {self.l}.'

    def kernel(self, X, Y, centered=False):
        K = cdist(X, Y, metric='minkowski', p=1)
        K = np.exp(-self.l*K)
        if centered:
            K = center_kernel(K)
        return K


class LinearKernel:
    def __init__(self):
        pass

    def kernel(self, X, Y, centered=False):
        K = X @ Y.T
        if centered:
            K = center_kernel(K)
        return K

    def __str__(self):
        return "Noyau linéaire."


class SigmoidKernel:
    def __init__(self, gamma=1, coef0=1):
        self.gamma = gamma
        self.coef0 = coef0

    def kernel(self, X, Y, centered=False):
        K = X @ Y.T
        K *= self.gamma
        K += self.coef0
        np.tanh(K, K)
        if centered:
            K = center_kernel(K)
        return K

    def __str__(self):
        return f"Noyau Sigmoid de paramètre gamma: {self.gamma}, et de coefficient: {self.coef0}."
