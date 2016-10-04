from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np


def otsu(src_path, dst_path):
    img = cv2.imread(src_path)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.max()
    hist_cumsum = hist.cumsum()

    indexes = np.arange(256)

    g_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p0, p1 = np.hsplit(hist, [i])
        w0, w1 = hist_cumsum[i], hist_cumsum[255] - hist_cumsum[i]
        ind0, ind1 = np.hsplit(indexes, [i])

        m0, m1 = np.sum(p0 * ind0) / w0, np.sum(p1 * ind1) / w1
        g0, g1 = np.sum(((ind0 - m0) ** 2) * p0) / w0, np.sum(((ind1 - m1) ** 2) * p1) / w1

        gt = g0 * w0 + g1 * w1

        if gt < g_min:
            g_min = gt
            thresh = i

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray > thresh] = 255
    gray[gray <= thresh] = 0
    cv2.imwrite(dst_path, gray)


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])
