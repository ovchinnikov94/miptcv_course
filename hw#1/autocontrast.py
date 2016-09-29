from __future__ import print_function
from sys import argv
import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def autocontrast(src_path, dst_path, white_perc, black_perc):
    img = cv2.imread(src_path)
    #cv2.imshow("start", img)
    b,g,r = cv2.split(img)
    maxB = np.max(b)
    maxG = np.max(g)
    maxR = np.max(r)
    minB = np.min(b)
    minG = np.min(g)
    minR = np.min(r)
    b -= minB
    g -= minG
    r -= minR
    b *= np.uint8(255.0 / (maxB - minB))
    g *= np.uint8(255.0 / (maxG - minG))
    r *= np.uint8(255.0 / (maxR - minR))
    result = cv2.merge((b, g, r))
    result[result > 255] = 255.0
    result[result < 0] = 0
    result[result > 255 - float(white_perc) * 255] = 255
    result[result < float(black_perc) * 255] = 0
    result.astype(np.uint8)
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    assert 0 <= float(argv[3]) < 1
    assert 0 <= float(argv[4]) < 1

    autocontrast(*argv[1:])
