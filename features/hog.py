# Reference Navneet Dalal and Bill Triggs https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
import numpy as np
from tqdm import tqdm


def hog_features(images, nbins=9):
    n = len(images)
    hist = np.zeros((n, 27, 4 * nbins))
    for k, image in tqdm(enumerate(images)):
        hist[k] = hog_features_image(image, nbins)
    return hist


def hog_features_image(image, nbins=9):
    hist = np.zeros((3, 9, 4 * nbins))
    for c in range(3):
        hist[c] = hog_features_channel(image[..., c], nbins)
    return hist.reshape(27, 4 * nbins)


def hog_features_channel(data, nbins=9):
    w = 180 / nbins
    n, m = data.shape
    hist = np.zeros((4, 4, nbins))  # une cellule = un block de 8x8, image de taille 32x32: 1 seul patch
    for i in range(0, n):
        for j in range(0, m):
            gx, gy = 0, 0
            if i > 0:
                gx -= data[i - 1, j]

            if i < n - 1:
                gx += data[i + 1, j]

            if j > 0:
                gy -= data[i, j - 1]

            if j < m - 1:
                gy += data[i, j + 1]

            if not (gx or gy):
                continue

            magnitude = np.sqrt(gx ** 2 + gy ** 2)

            angle = np.arctan2(gx, gy)
            angle = (angle + np.pi) / (2 * np.pi / nbins)

            bin = int(np.floor(angle))

            if bin == nbins:
                bin = 0
                angle = 0

            closest_bin = bin

            if bin == 0:
                if angle < 0.5:
                    closest_bin_2 = nbins - 1
                else:
                    closest_bin_2 = 1
            elif bin == nbins - 1:
                if angle < nbins - 0.5:
                    closest_bin_2 = nbins - 2
                else:
                    closest_bin_2 = 0
            else:
                if angle < bin + 0.5:
                    closest_bin_2 = bin - 1
                else:
                    closest_bin_2 = bin + 1

            if angle < bin + 0.5:
                closest_bin_2_dist = angle - (bin - 0.5)
            else:
                closest_bin_2_dist = (bin + 1.5) - angle

            r = closest_bin_2_dist
            hist[i // 8, j // 8, closest_bin] += r * magnitude
            hist[i // 8, j // 8, closest_bin_2] += (1 - r) * magnitude

    concat_hist = np.zeros((3, 3, 4 * nbins))
    for i in range(3):
        for j in range(3):
            concat_hist[i, j] = hist[i:i + 2, j:j + 2].flatten() / np.linalg.norm(hist[i:i + 2, j:j + 2])

    return concat_hist.reshape(9, 4 * nbins)


if __name__ == '__main__':
    x = np.random.normal(size=(4, 32, 32, 3))
    print(hog_features(x))
