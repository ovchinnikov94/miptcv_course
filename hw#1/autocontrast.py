from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np


def autocontrast(src_path, dst_path, white_perc, black_perc):
    img = cv2.imread(src_path)
    #cv2.imshow("start", img)
    b,g,r = cv2.split(img)
    maxB, maxG, maxR, minB, minG, minR = np.max(b), np.max(g), np.max(r), np.min(b), np.min(g), np.min(r)
    b[b > maxB - maxB * float(white_perc)] = 255;
    b[b <= minB + float(black_perc) * (maxB - minB)] = 0

    g[g > maxG - maxG * float(white_perc)] = 255
    g[g <= minG + float(black_perc) * (maxG - minG)] = 0

    r[r > maxR - maxR * float(white_perc)] = 255
    r[r <= minR + float(black_perc) * (maxR - minR)] = 0

    b -= minB
    g -= minG
    r -= minR
    b *= np.uint8(255.0 / (maxB - minB))
    g *= np.uint8(255.0 / (maxG - minG))
    r *= np.uint8(255.0 / (maxR - minR))

    result = cv2.merge((b, g, r))
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
