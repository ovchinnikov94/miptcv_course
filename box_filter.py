from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np


def box_filter(src_path, dst_path, w, h):
    img = cv2.imread(src_path)
    width, height, _ = img.shape
    integral_sum = cv2.integral(img)
    for i in range(width):
        for j in range(height):
            w1 = max(i + w % 2 - w / 2, 0)
            h1 = max(j + h % 2 - h / 2, 0)
            w2 = min(i + w % 2 + w / 2, width - 1)
            h2 = min(j + h % 2 + h / 2, height - 1)
            cur_sum = integral_sum[w1][h1] + integral_sum[w2][h2] - integral_sum[w1][h2] - integral_sum[w2][h1]
            filter_size = (w2 - w1) * (h2 - h1)
            img.itemset((i, j, 0), np.uint8(float(cur_sum[0]) / filter_size))
            img.itemset((i, j, 1), np.uint8(float(cur_sum[1]) / filter_size))
            img.itemset((i, j, 2), np.uint8(float(cur_sum[2]) / filter_size))
    cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_filter(*argv[1:])
