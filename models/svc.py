# Simple SVM
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from tqdm import tqdm

from itertools import combinations


class SVC:

    def __init__(self, C=1, kernel=None, tol=1e-5):
        self.kernel = kernel
        self.C = C

        self.X = None
        self.y = None

        self.alpha = None
        self.b = None

        self.scale = None

        self.tol = tol

        self.X_multi = None

        self.alpha_multi = None
        self.b_multi = None

        self.scale_multi = None

        self.nb_class = None

    def fit(self, X, y):
        self.b = 0

        N = len(y)
        hXX = self.kernel(X, X)
        G = np.einsum('ij,i,j->ij', hXX, y, y)
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))

        def loss(alpha):
            return -alpha.sum() + 0.5 * alpha.dot(alpha.dot(G))

        def grad_loss(alpha):
            return -np.ones_like(alpha) + alpha.dot(G)

        fun_eq = lambda alpha: np.dot(alpha, y)
        jac_eq = lambda alpha: y
        fun_ineq = lambda alpha: b - np.dot(A, alpha)
        jac_ineq = lambda alpha: -A

        constraints = ({'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq',
                        'fun': fun_ineq,
                        'jac': jac_ineq})

        optRes = minimize(fun=lambda alpha: loss(alpha),
                          x0=np.random.normal(size=N),
                          method='SLSQP',
                          jac=lambda alpha: grad_loss(alpha),
                          constraints=constraints)
        self.alpha = optRes.x

        margin_pointsIndices = self.alpha > self.tol
        boundaryIndices = (self.alpha > self.tol) * (self.C - self.alpha > self.tol)
        self.X = X[margin_pointsIndices]
        self.y = y[margin_pointsIndices]
        self.alpha = y[margin_pointsIndices] * self.alpha[margin_pointsIndices]
        self.b = y[boundaryIndices][0] - \
                    self.separating_function(np.expand_dims(X[boundaryIndices][0], axis=0))

        self._platt_fit()

    def separating_function(self, x):
        x1 = self.kernel(self.X, x)
        return np.einsum('ij,i->j', x1, self.alpha) + self.b

    def predict(self, X):
        d = self.separating_function(X)
        return 2 * (d + self.b > 0) - 1

    def _platt_fit(self):
        mylog = lambda x: 0 if x == 0 else np.log(x)
        sigmoid = lambda x, param1, param2: expit(-(x * param1 + param2))

        out = self.separating_function(self.X)
        L = self.y
        target = L == 1
        prior1 = np.float64(np.sum(target))
        prior0 = len(target) - prior1

        A = 0
        B = np.log((prior0 + 1) / (prior1 + 1))
        hiTarget = (prior1 + 1) / (prior1 + 2)
        loTarget = 1 / (prior0 + 2)
        labda = 1e-3
        olderr = 1e300

        pp = np.ones(out.shape) * (prior1 + 1) / (prior0 + prior1 + 2)

        T = np.zeros(target.shape)
        for _ in range(1, 200):
            a, b, c, d, e = 0, 0, 0, 0, 0
            for i in range(len(out)):
                if target[i]:
                    t = hiTarget
                    T[i] = t
                else:
                    t = loTarget
                    T[i] = t
                d1, d2 = pp[i] - t, pp[i] * (1 - pp[i])

                a += out[i] * out[i] * d2
                b += d2
                c += out[i] * d2
                d += out[i] * d1
                e += d1
            if abs(d) < 1e-9 and abs(e) < 1e-9:
                break
            oldA, oldB = A, B
            err, count = 0, 0
            while True:
                det = (a + labda) * (b + labda) - c * c
                if det == 0:
                    labda *= 10
                A = oldA + ((b + labda) * d - c * e) / det
                B = oldB + ((a + labda) * e - c * d) / det
                err = 0
                for i in range(len(out)):
                    p = sigmoid(out[i], A, B)
                    pp[i] = p
                    t = T[i]
                    err -= t * mylog(p) + (1 - t) * mylog(1 - p)
                if err < olderr * (1 + 1e-7):
                    labda *= 0.1
                    break
                labda *= 10
                if labda > 1e6:
                    break
            diff = err - olderr
            scale = 0.5 * (err + olderr + 1)
            if -1e-3 * scale < diff < 1e-7 * scale:
                count += 1
            else:
                count = 0
            olderr = err
            if count == 3:
                break
        self.scale = lambda x: sigmoid(x, A, B)

    def fit_multiclass(self, X, y):
        nb_class = np.max(y) + 1
        self.nb_class = nb_class
        alpha_multi, X_multi, b_multi = {}, {}, {}
        scale_multi = {}

        for i, j in tqdm(combinations(range(nb_class), 2)):
            ind_i, ind_j = y == i, y == j
            y_ij = y[ind_i + ind_j]
            y_ij[y_ij == i], y_ij[y_ij == j] = -1, 1
            X_ij = X[ind_i + ind_j]
            self.fit(X_ij, y_ij)
            alpha_multi[(i, j)] = self.alpha
            b_multi[(i, j)] = self.b
            X_multi[(i, j)] = self.X
            scale_multi[(i, j)] = self.scale

        self.X_multi = X_multi
        self.scale_multi = scale_multi
        self.alpha_multi = alpha_multi
        self.b_multi = b_multi

    def predict_multiclass(self, X):
        n = len(X)
        prediction = np.array([[[0., 0.] for _ in range(self.nb_class)] for _ in range(n)])

        for i, j in combinations(range(self.nb_class), 2):
            X_ij, alpha_ij, b_ij = self.X_multi[(i, j)], self.alpha_multi[(i, j)], self.b_multi[(i, j)]
            scale_ij = self.scale_multi[(i, j)]

            predict_labels_ij = np.einsum('ij,i->j', self.kernel(X_ij, X), alpha_ij) + b_ij
            predict_labels_ij = scale_ij(predict_labels_ij)
            for k in range(n):
                prediction[k][j] += np.array([predict_labels_ij[k], 1])
                prediction[k][i] += np.array([1 - predict_labels_ij[k], 1])

        final_prediction = np.zeros(n)
        for k, probs in enumerate(prediction):
            proba = probs[:, 0] / probs[:, 1]
            final_prediction[k] = np.argmax(proba)

        return final_prediction
