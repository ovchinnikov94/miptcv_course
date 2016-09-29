from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np


def gamma_correction(src_path, dst_path, a, b):
    img = cv2.imread(src_path)
    table = np.array([(np.float64(a) * ((i / 255.0) ** np.float64(b))) * 255 for i in np.arange(0, 256)])\
        .astype("uint8")
    result = cv2.LUT(img, table)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])

    gamma_correction(*argv[1:])
