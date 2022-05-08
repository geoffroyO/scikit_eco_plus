# Logistic regression
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from models import Model

logistic_loss = lambda x : -log_expit(x)
logistic_loss_1 = lambda x : expit(x) - 1
logistic_loss_2 = lambda x : np.multiply(expit(x), 1 - expit(x))

class KLR(Model):
    def __init__(self, lbd=1, kernel='linear', **kwargs):
        super().__init__(kernel, **kwargs)
        self.lbd = lbd
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features = n_features

        K = self.kernel(X, X)
        
        obj = lambda alpha : np.mean(logistic_loss(np.multiply(y, K@alpha))) + self.lbd*0.5*alpha.T@K@alpha

        p = lambda alpha : np.diag(logistic_loss_1(np.multiply(y, K@alpha)))
        jac = lambda alpha : (1/n_samples)*K@p(alpha)@y + self.lbd*K@alpha

        w = lambda alpha : np.diag(logistic_loss_2(np.multiply(y, K@alpha)))
        hess = lambda alpha : (1/n_samples)*K@w(alpha)@K + self.lbd*K

        alpha_0 = np.random.normal(size=y.shape)
        
        res = minimize(obj, alpha_0, method='Newton-CG', jac=jac, hess=hess)
        self.alpha = res.x
        self.X_model = X

    def predict(self, X):
        _, n_features = X.shape
        if n_features != self.n_features:
            raise ValueError(
                "Erreur nombre de features différent \
                    du modèle"
            )
        predict_labels = np.dot(self.kernel(X, self.X_model), self.alpha)
        predict_labels[predict_labels <= 0] = -1
        predict_labels[predict_labels >= 0] = 1
        return predict_labels
