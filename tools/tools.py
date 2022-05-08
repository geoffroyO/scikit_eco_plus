import numpy as np
import cv2
import pandas as pd


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
    if shuffle:
        np.random.seed(random_state)
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]

    n_sample = len(y)
    test_size = int(n_sample * test_size)

    X_test, y_test = X[:test_size], y[:test_size]
    X_train, y_train = X[test_size:], y[test_size:]

    return X_train, y_train, X_test, y_test


def classification_report(y_true, y_pred):
    class_label = np.unique(y_true)
    print('         precision       recall      f1-score')
    for c in class_label:
        vp, fp, vn, fn = 0, 0, 0, 0
        for true, pred in zip(y_true, y_pred):
            if pred == c:
                if true == c:
                    vp += 1
                else:
                    fp += 1
            else:
                if true == c:
                    fn += 1
                else:
                    vn += 1
        if vp + fp == 0:
            prec = np.nan
        else:
            prec = vp / (vp + fp)
        if vp + fn == 0:
            recall = np.nan
        else:
            recall = vp / (vp + fn)
        f1_score = 2 * (prec * recall) / (prec + recall)
        print('Class {}     {:.2f}           {:.2f}        {:.2f}'.format(c, prec, recall, f1_score))
    print('Average accuracy   {:.2f}'.format(np.mean(y_pred == y_true)))


def confusion_matrix(actual, predicted):
    classes = np.unique(actual)

    confmat = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
            confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat


def data_loader(path):
    Xtr = np.array(pd.read_csv(path + 'Xtr.csv', header=None, sep=',', usecols=range(3072))).reshape((5000, 3, 32, 32))
    Xtr = np.swapaxes(Xtr, 1, 2)
    Xtr = np.swapaxes(Xtr, 2, 3)
    Xtr = cv2.normalize(Xtr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    Xtr = Xtr.astype(np.uint8) / 255

    Ytr = np.array(pd.read_csv(path + 'Ytr.csv', sep=',', usecols=[1])).squeeze()

    Xte = np.array(pd.read_csv(path + 'Xte.csv', header=None, sep=',', usecols=range(3072))).reshape((2000, 3, 32, 32))
    Xte = np.swapaxes(Xte, 1, 2)
    Xte = np.swapaxes(Xte, 2, 3)
    Xte = cv2.normalize(Xte, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    Xte = Xte.astype(np.uint8) / 255
    return Xtr, Ytr, Xte


def data_augmentation(Xtr, Ytr):
    Xtr_aug = []
    Ytr_aug = []

    for image, label in zip(Xtr, Ytr):
        fliph_im = np.fliplr(image)
        Xtr_aug += [image, fliph_im]
        Ytr_aug += [label] * 2

    Xtr_aug = np.array(Xtr_aug)
    Ytr_aug = np.array(Ytr_aug)

    Xtr_jit = []
    Ytr_jit = []
    for image, label in zip(Xtr_aug, Ytr_aug):
        sigma = np.random.randint(1, 5) / 100
        noisy_image = np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)
        Xtr_jit += [image, noisy_image]
        Ytr_jit += [label] * 2

    Xtr_jit = np.array(Xtr_jit)
    Ytr_jit = np.array(Ytr_jit)

    return Xtr_jit, Ytr_jit

    # return Xtr_aug, Ytr_aug
