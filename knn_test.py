# coding=utf-8
# 2018/6/27 15:07

import numpy as np
from knn_digits import classify


def create_data_set():
    _group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    _labels = ['A', 'A', 'B', 'B']
    return _group, _labels


if __name__ == '__main__':
    group, label = create_data_set()
    test1 = [0.3, 0.2]
    result1 = classify(test1, group, label)
    test2 = [0.8, 0.6]
    result2 = classify(test2, group, label)

    print('%s 属于分类 %s' % (test1, result1))
    print('%s 属于分类 %s' % (test2, result2))
