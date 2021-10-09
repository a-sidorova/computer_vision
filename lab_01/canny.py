import math
import numpy as np
from utils import sign, is_correct_idx
from filters import RGB2Gray, GaussFilter, Sobel, DoubleTresholding


class Canny:
    @staticmethod
    def get(img):
        gray_img = RGB2Gray.calculate(img)
        gauss_img = GaussFilter.calculate(gray_img)
        sobel_img, grad_img = Sobel.calculate(gauss_img)
        non_max_suppression_img = NMS(sobel_img, grad_img)
        result_img = DoubleTresholding.calculate(non_max_suppression_img, 0.5, 0.7)
        return result_img


def NMS(sobel, grad):
    h, w = sobel.shape
    res = np.zeros(shape=(h, w), dtype='float')
    for y in range(h - 1):
        for x in range(w - 1):
            if grad[y, x] == -1:
                continue
            dx = sign(math.cos(grad[y, x]))
            dy = -1 * sign(math.sin(grad[y, x]))
            if is_correct_idx(w, h, x + dx, y + dy):
                if sobel[y + dy, x + dx] <= sobel[y, x]:
                    res[y + dy, x + dx] = 0
            if is_correct_idx(w, h, x - dx, y - dy):
                if sobel[y - dy, x - dx] <= sobel[y, x]:
                    res[y - dy, x - dx] = 0
            res[y, x] = sobel[y, x]
    return res

