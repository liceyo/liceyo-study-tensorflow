# 绘制树形图
import matplotlib.pyplot as plt
import trees

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 获取树的叶子结点个数（确定图的宽度）
def get_leaf_num(tree):
    leaf_num = 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == "dict":
            leaf_num += get_leaf_num(next_dict[key])
        else:
            leaf_num += 1
    return leaf_num


# 获取数的深度（确定图的高度）
def get_tree_depth(tree):
    depth = 0
    first_key = list(tree.keys())[0]
    next_dict = tree[first_key]
    for key in next_dict.keys():
        if type(next_dict[key]).__name__ == "dict":
            thisdepth = 1 + get_tree_depth(next_dict[key])
        else:
            thisdepth = 1
        if thisdepth > depth: depth = thisdepth
    return depth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 在父子节点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = get_leaf_num(myTree)
    depth = get_tree_depth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decision_node)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(get_leaf_num(inTree))
    plotTree.totalD = float(get_tree_depth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    data_set, labels = trees.create_test_data_set()
    tree = trees.create_tree(data_set, labels)
    createPlot(tree)

