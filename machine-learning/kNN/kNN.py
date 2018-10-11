# k-近邻算法
from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    # 行数
    dataSetSize = data_set.shape[0]
    # 距离计算
    # 扩展in_x为data_set长度的数组，求差
    diffMat = tile(in_x, (dataSetSize, 1)) - data_set
    # 平方
    sqDiffMat = diffMat ** 2
    # 平方和 按行求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方获取到距离
    distances = sqDistances ** 0.5
    # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    sortedDistIndices = distances.argsort()
    print(sortedDistIndices)
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteILabel = labels[sortedDistIndices[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount


group, labels = create_data_set()
result = classify0([0.9, 1.2], group, labels, 3)
print(result)
