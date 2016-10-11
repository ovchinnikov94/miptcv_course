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

    g_max = 0
    thresh = -1
    for i in range(1, 256):
        p0, p1 = np.hsplit(hist, [i])
        w0, w1 = hist_cumsum[i], hist_cumsum[255] - hist_cumsum[i]
        ind0, ind1 = np.hsplit(indexes, [i])
        if w0 != 0:
            m0 = np.sum(p0 * ind0) / w0
        else:
            m0 = 0
        if w1 != 0:
            m1 = np.sum(p1 * ind1) / w1
        else:
            m1 = 0

        g_between = w0 * w1 * (np.sum((m1 - m0) ** 2))

        if g_between > g_max:
            g_max = g_between
            thresh = i

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray > thresh] = 255
    gray[gray <= thresh] = 0
    cv2.imwrite(dst_path, gray)


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])
