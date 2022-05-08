import numpy as np
from numpy.linalg import inv
from scikit_eco_plus.models.models import Model


class KernelRidge(Model):
    def __init__(self, lbd=1, kernel='linear', **kwargs):
        super().__init__(kernel, **kwargs)
        self.alpha = None
        self.n_features = None
        self.X_model = None
        self.lbd = lbd
    
    def fit(self, X, y):
        self.X_model = X
        n_samples, n_features = X.shape
        self.n_features = n_features

        K = self.kernel(X, X)
        self.alpha = np.dot(inv(K + n_samples*self.lbd*np.eye(n_samples)), y)
    
    def predict(self, X):
        _, n_features = X.shape
        if n_features != self.n_features:
            raise ValueError(
                "Erreur nombre de features différent \
                    du modèle"
            )
        predict_labels = np.dot(self.kernel(X, self.X_model), self.alpha)
        return predict_labels
