# coding=utf-8
# 2018/6/27 15:08

import numpy as np
import operator
from os import listdir


def classify(inX, dataSet, labels, k=3):
    """
    knn分类器
    :param inX: 要计算的矩阵
    :param dataSet: 训练集矩阵
    :param labels: 训练集标签
    :param k: k值
    :return:
    """
    dataSetSize = dataSet.shape[0]

    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    distances = (diffMat ** 2).sum(axis=1) ** 0.5

    sorted_distances = distances.argsort()
    class_count = {}

    for i in range(k):
        vort_label = labels[sorted_distances[i]]
        class_count[vort_label] = class_count.get(vort_label, 0) + 1

    sotred_distances = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sotred_distances[0][0]


def img2vector(filename):
    return_vect = np.zeros([1, 1024])

    fr = open(filename)

    for i in range(32):
        line = fr.readline()

        for j in range(32):
            return_vect[0, 32 * i + j] = int(line[j])

    return return_vect


def digits():
    # 样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    # 训练集标签
    hwLabels = []
    # 训练集
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = trainingFileList[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hwLabels.append(class_num_str)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % file_name_str)

    # 循环读取测试数据
    testFileList = listdir('digits/testDigits')

    # 错误率初始化
    error_count = 0
    mTest = len(testFileList)
    for i in range(mTest):
        file_name_str = testFileList[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])

        # 提取向量
        vector_test = img2vector('digits/testDigits/%s' % file_name_str)

        # 对向量进行分类
        classify_result = classify(vector_test, trainingMat, hwLabels, 3)

        # 打印KNN分类结果和真实分类
        print("the classifier came back with: %d, the real answer is: %d" % (classify_result, class_num_str))

        if classify_result != class_num_str:
            error_count += 1

    # 结果汇总
    print('train size: %s, test size: %s' % (m, mTest))
    print("the total number of errors is: %d" % error_count)
    print("the total error rate is: %f" % (error_count / float(mTest)))


if __name__ == '__main__':
    digits()
