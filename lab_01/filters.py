import abc
import math
import numpy as np


class Filter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, img):
        pass


class RGB2Gray(Filter):
    @staticmethod
    def calculate(img):
        h, w, _ = img.shape
        res = np.zeros(shape=(h, w), dtype='uint8')
        for x in range(w):
            for y in range(h):
                b = img[y, x, 0]
                g = img[y, x, 1]
                r = img[y, x, 2]
                s = 0.2952 * r + 0.5547 * g + 0.148 * b
                res[y, x] = s
        return res


class GaussFilter(Filter):
    @staticmethod
    def calculate(img, radius=3, sigma=1):
        h, w = img.shape
        result = np.zeros(shape=(h, w), dtype='uint8')

        def create_gaussian_kernel():
            norm = 0
            size = 2 * radius + 1
            kernel_ = np.zeros(shape=(size, size), dtype='float32')
            for i in range(radius, radius + 1):
                for j in range(-radius, radius + 1):
                    kernel_[i + radius, j + radius] = \
                        (math.exp((-1) * (i * i + j * j) / (sigma * sigma)))
                    norm += kernel_[i + radius, j + radius]
            for i in range(size):
                for j in range(size):
                    kernel_[i, j] = kernel_[i, j] / norm
            return kernel_

        def calculate_new_color(x_, y_):
            g = 0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    idx = max(min(x_ + i, w - 1), 0)
                    idy = max(min(y_ + j, h - 1), 0)
                    color = img[idy, idx]
                    g += color * kernel[i, j]
            return int(max(min(g, 255), 0))

        kernel = create_gaussian_kernel()
        for x in range(w):
            for y in range(h):
                result[y, x] = calculate_new_color(x, y)
        return result


class Sobel(Filter):
    @staticmethod
    def calculate(img):
        h, w = img.shape
        result = np.zeros(shape=(h, w), dtype='float')
        result_grad = np.zeros(shape=(h, w), dtype='float')

        m_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        m_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                g_x = 0
                g_y = 0
                for p in range(-1, 2):
                    for q in range(-1, 2):
                        g_x += img[y + p, x + q] * m_x[p + 1, q + 1]
                        g_y += img[y + p, x + q] * m_y[p + 1, q + 1]
                g = math.sqrt(g_x * g_x + g_y * g_y)
                t = int(math.atan2(g_x, g_y) / (math.pi / 4)) * (math.pi / 4) - math.pi / 2 if g != 0 else -1
                result[y, x] = g
                result_grad[y, x] = t
        return result, result_grad


class DoubleTresholding(Filter):
    @staticmethod
    def calculate(img, low, high):
        down = int(255 * low)
        up = int(255 * high)
        h, w = img.shape
        res = np.zeros(shape=(h, w), dtype='uint8')
        for y in range(h - 1):
            for x in range(w - 1):
                res[y, x] = 255 if img[y, x] >= up else (0 if img[y, x] <= down else 127)
        return res
