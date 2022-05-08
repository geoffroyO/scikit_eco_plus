# SVM + SMO + platt scaling
from itertools import combinations

import numpy as np
from scipy.special import expit
from tqdm import tqdm

from scikit_eco_plus.utils.kernels import GaussianKernel, LinearKernel


class SVC:
    def __init__(self, C, kernel, tol=1e-5, eps=1e-2, prec=1e-4, max_iter=50):
        self.C = C
        self.kernel = kernel

        self.tol = tol
        self.eps = eps
        self.prec = prec
        self.max_iter = max_iter

        self.X = None
        self.y = None

        self.alpha = None
        self.b = None

        self.errors = None

        self.scale = None

        self.n = None
        self.K = None

    def fit(self, X, y):
        self.X, self.y, self.n = X, y, len(X)
        self.alpha = np.zeros(self.n)
        self.b = 0
        self.errors = self.separating_function(self.X) - self.y
        self.K = self.kernel(self.X, self.X)

        numChanged = 0
        examineAll = 1
        it = 0
        prev_obj, cur_obj = np.inf, self.dual_obj(self.alpha)
        while ((numChanged > 0) or (examineAll == 1)) \
                and (it < self.max_iter) \
                and (np.abs(prev_obj - cur_obj) >= self.prec):
            numChanged = 0
            if examineAll:
                for i in range(self.n):
                    numChanged += self.examine_example(i)
            else:
                for i in np.where((self.alpha != 0) & (self.alpha != self.C))[0]:
                    numChanged += self.examine_example(i)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
            it += 1
            prev_obj, cur_obj = cur_obj, self.dual_obj(self.alpha)
        ind = self.alpha > self.tol
        self.alpha = self.alpha[ind]
        self.X = self.X[ind]
        self.y = self.y[ind]

        self._platt_scale()

    def examine_example(self, i2):
        y2 = self.y[i2]
        alph2 = self.alpha[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):

            if len(self.alpha[(self.alpha != 0) & (self.alpha != self.C)]) > 1:
                if self.errors[i2] > 0:
                    i1 = np.argmin(self.errors)
                else:
                    i1 = np.argmax(self.errors)
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1

            for i1 in np.roll(np.where((self.alpha != 0) & (self.alpha != self.C))[0],
                              np.random.choice(np.arange(self.n))):
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1

            for i1 in np.roll(np.arange(self.n), np.random.choice(np.arange(self.n))):
                step_result = self.take_step(i1, i2)
                if step_result:
                    return 1

        return 0

    def take_step(self, i1, i2):
        if i1 == i2:
            return 0

        alph1 = self.alpha[i1]
        alph2 = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2

        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)
        else:
            L = max(0, alph1 + alph2 - self.C)
            H = min(self.C, alph1 + alph2)
        if L == H:
            return 0

        xi1, xi2 = self.X[i1][np.newaxis], self.X[i2][np.newaxis]
        k11, k12, k22 = self.kernel(xi1, xi1)[0, 0], self.kernel(xi1, xi2)[0, 0], self.kernel(xi2, xi2)[0, 0]

        eta = 2 * k12 - k11 - k22

        if eta < - 1e-3:
            a2 = alph2 - y2 * (E1 - E2) / eta
            if L < a2 < H:
                a2 = a2
            elif a2 <= L:
                a2 = L
            elif a2 >= H:
                a2 = H

        else:
            alpha_tmp = np.copy(self.alpha)

            alpha_tmp[i2] = L
            Lobj = self.dual_obj(alpha_tmp)

            alpha_tmp[i2] = H
            Hobj = self.dual_obj(alpha_tmp)
            if Lobj < (Hobj - self.eps):
                a2 = L
            elif Lobj > (Hobj + self.eps):
                a2 = H
            else:
                a2 = alph2

        if a2 < self.tol:
            a2 = 0
        elif a2 > (self.C - self.tol):
            a2 = self.C

        if np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return 0

        a1 = alph1 + s * (alph2 - a2)

        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b

        if 0 < a1 < self.C:
            b_new = b1
        elif 0 < a2 < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) * 0.5

        self.alpha[i1] = a1
        self.alpha[i2] = a2

        for index, alph in zip([i1, i2], [a1, a2]):
            if 0 < alph < self.C:
                self.errors[index] = 0

        non_opt = [n for n in range(self.n) if (n != i1 and n != i2)]
        self.errors[non_opt] = self.errors[non_opt] + \
                               y1 * (a1 - alph1) * self.kernel(xi1, self.X[non_opt]) + \
                               y2 * (a2 - alph2) * self.kernel(xi2, self.X[non_opt]) + \
                               self.b - b_new

        self.b = b_new

        return 1

    def _platt_scale(self):
        log = lambda x: 0 if x == 0 else np.log(x)
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
                    err -= t * log(p) + (1 - t) * log(1 - p)
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

    def separating_function(self, x):
        tmp = self.alpha * self.y
        return np.dot(tmp, self.kernel(self.X, x)) - self.b

    def dual_obj(self, alpha_tmp):
        tmp = self.y * alpha_tmp
        return 0.5 * np.sum(np.dot(tmp, np.dot(self.K, tmp))) - np.sum(alpha_tmp)

    def predict(self, x):
        d = self.separating_function(x) > 0
        return 2 * d - 1


class SVCmulti:
    def __init__(self, C, kernel, tol=1e-5, eps=1e-2, prec=1e-4, max_iter=100):
        self.C = C
        self.kernel = kernel

        self.tol = tol
        self.eps = eps
        self.prec = prec
        self.max_iter = max_iter

        self.nb_class = None

        self.X_multi = None
        self.y_multi = None
        self.alpha_multi = None
        self.b_multi = None
        self.scale_multi = None

    def fit_multiclass(self, X, y):
        nb_class = np.max(y) + 1
        self.nb_class = nb_class
        alpha_multi, X_multi, b_multi, y_multi = {}, {}, {}, {}
        scale_multi = {}

        for i, j in tqdm(combinations(range(nb_class), 2)):
            ind_i, ind_j = y == i, y == j
            y_ij = y[ind_i + ind_j]
            y_ij[y_ij == i], y_ij[y_ij == j] = -1, 1
            X_ij = X[ind_i + ind_j]
            svc = SVC(self.C, self.kernel, self.tol, self.eps, self.prec, self.max_iter)
            svc.fit(X_ij, y_ij)
            alpha_multi[(i, j)] = svc.alpha
            b_multi[(i, j)] = svc.b
            X_multi[(i, j)] = svc.X
            y_multi[(i, j)] = svc.y
            scale_multi[(i, j)] = svc.scale

        self.X_multi = X_multi
        self.y_multi = y_multi
        self.scale_multi = scale_multi
        self.alpha_multi = alpha_multi
        self.b_multi = b_multi

    def predict_multiclass(self, X):
        n = len(X)
        prediction = np.array([[[0., 0.] for _ in range(self.nb_class)] for _ in range(n)])

        for i, j in combinations(range(self.nb_class), 2):
            X_ij, y_ij, alpha_ij, b_ij = self.X_multi[(i, j)], self.y_multi[(i, j)], self.alpha_multi[(i, j)], self.b_multi[(i, j)]
            scale_ij = self.scale_multi[(i, j)]
            tmp = alpha_ij * y_ij
            predict_labels_ij = np.dot(tmp, self.kernel(X_ij, X)) - b_ij
            predict_labels_ij = scale_ij(predict_labels_ij)
            for k in range(n):
                prediction[k][j] += np.array([predict_labels_ij[k], 1])
                prediction[k][i] += np.array([1 - predict_labels_ij[k], 1])

        final_prediction = np.zeros(n)
        for k, probs in enumerate(prediction):
            proba = probs[:, 0] / probs[:, 1]
            final_prediction[k] = np.argmax(proba)

        return final_prediction


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()

    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    C = 1

    model = SVCmulti(C, LinearKernel().kernel)
    model.fit_multiclass(X_train, y_train)

    print((model.predict_multiclass(X_test) == y_test).mean())
