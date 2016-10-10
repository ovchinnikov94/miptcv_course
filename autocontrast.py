from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np


def autocontrast(src_path, dst_path, white_perc, black_perc):
    img = cv2.imread(src_path, 0)

    '''Make the most white_perc% bright pixels to be 255 and the most black_perc% dark pixels to be 0'''
    max_value, min_value = np.max(img), np.min(img)
    width, height = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    '''Find high boundary'''
    high_bound = max_value
    while float(white_perc) * width * height > np.sum(hist[high_bound:]):
        high_bound -= 1
    '''Find low boundary'''
    low_bound = min_value
    while float(black_perc) * width * height > np.sum(hist[:low_bound]):
        low_bound += 1

    img[img >= high_bound] = 255
    img[img <= low_bound] = 0

    '''Do autocontrast'''
    result = img.astype(float)
    result -= min_value
    result *= 255.0 / (max_value - min_value)
    result[result > 255] = 255
    result[result < 0] = 0
    result = result.astype('uint8')

    cv2.imwrite(dst_path, np.hstack([result]))


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    assert 0 <= float(argv[3]) < 1
    assert 0 <= float(argv[4]) < 1

    autocontrast(*argv[1:])
