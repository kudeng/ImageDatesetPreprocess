from resources import paths
import random
from matplotlib import pyplot as plt


def read_file(csvf):
    """
    读取csv格式的mnist数据
    :param csvf: mnist训练或测试集
    :return: 数据及标签
    """
    with open(csvf, "r") as f1:
        data = f1.readlines()
    X = []
    y = []
    for line in data:
        elements = line.split(',')
        y.append(int(elements[0]))
        for e in elements[1:]:
            X.append(int(e))
    return X, y


file = paths.PRODUCT_PATH + "svhn_train.csv"
(x, y) = read_file(file)
print(len(y))

for i in range(8):
    num = int(random.random())
    img = x[num]
    label = y[num]
    print(label)
    plt.imshow(img)
    plt.show()