from sklearn import datasets
import numpy as np
import trees
import treePlotter


def create_digit_tree():
    digits = datasets.load_digits()
    d_data = digits.data
    d_target = digits.target
    labels = [i for i in range(len(d_target))]
    data_set = np.append(d_data, np.array([d_target]).T, axis=1).tolist()
    tree = trees.create_tree(data_set, labels)
    print(tree)
    # treePlotter.createPlot(tree)
    trees.store_tree(tree, "digit_tree.txt")


if __name__ == '__main__':
    tree = trees.grab_tree("digit_tree.txt")
    print(tree)
