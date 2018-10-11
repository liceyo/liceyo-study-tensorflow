import operator
from math import log


# ID3算法

# 计算数据集的香农熵,度量数据集的无序程度
def calc_shannon_ent(data_set):
    numEntries = len(data_set)
    labelCounts = {}
    for featVec in data_set:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 生成数据
def create_test_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no'],
                [2, 0, 't'],
                [2, 1, 't']]
    return data_set, ['no surfacing', 'flippers']


# 划分数据集
def split_data_set(data_set, axis, value):
    # 参数：
    # 待划分的数据集
    # 划分数据集的特征（即数据集的列号）
    # 需要返回的特征的值（即axis对应的值）
    retDataSet = []
    for featVec in data_set:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def choose_best_feature_to_split(data_set):
    numFeatures = len(data_set[0]) - 1
    baseEntropy = calc_shannon_ent(data_set)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in data_set]
        uniqueSet = set(featList)
        # 计算该种划分方式的熵
        newEntropy = 0.0
        for value in uniqueSet:
            subDataSet = split_data_set(data_set, i, value)
            prob = len(subDataSet) / float(len(data_set))
            newEntropy += prob * calc_shannon_ent(subDataSet)
        # 计算最好的信息增益
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 投票选出最多的标签
def majority_cnt(class_list):
    classCount = {}
    for vote in class_list:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def create_tree(data_set, labels):
    # 获取数据集的所有类标签
    classList = [example[-1] for example in data_set]
    # 递归结束条件一：所有类标签一致
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归结束条件二：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    if len(data_set[0]) == 1:
        return majority_cnt(classList)
    # 选择最好的数据集划分方式
    bestFeat = choose_best_feature_to_split(data_set)
    # 最好的数据集划分方式标签
    bestFeatLabel = labels[bestFeat]
    subLabels = labels[:]
    # 创建空树
    myTree = {bestFeatLabel: {}}
    # 删除标签
    del (subLabels[bestFeat])
    # 获取数据划分的value
    featValues = [example[bestFeat] for example in data_set]
    uniqueSet = set(featValues)
    # 递归构建树
    for value in uniqueSet:
        myTree[bestFeatLabel][value] = create_tree(split_data_set(data_set, bestFeat, value), subLabels)
    return myTree


# 决策树的分类函数
def classify(input_tree, feat_labels, test_vec):
    firstStr = list(input_tree.keys())[0]
    secondDict = input_tree[firstStr]
    featIndex = feat_labels.index(firstStr)
    classLabel = 'unknown'
    for key in secondDict.keys():
        if test_vec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], feat_labels, test_vec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 存储树
def store_tree(input_tree, filename):
    import pickle
    fw = open(filename, "wb+")
    pickle.dump(input_tree, fw)
    fw.close()


# 读取树
def grab_tree(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr, encoding="UTF-8")


if __name__ == '__main__':
    dataSet, labels = create_test_data_set()
    tree = create_tree(dataSet, labels)
    print(tree)
    filename = 'tree.txt'
    store_tree(tree, filename)
    print(classify(tree, labels, [0, 1]))
    print(classify(tree, labels, [1, 0]))
    print(classify(grab_tree(filename), labels, [1, 1]))
