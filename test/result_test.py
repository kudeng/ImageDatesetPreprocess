from resources import paths
from resources import nums
import numpy as np
import random
from matplotlib import pyplot as plt
import math

file = paths.PRODUCT_PATH + 'cifar10_train.npy'
# file = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/out/mnist_svhn_28281.npy'
data = np.load(file)
y = data[:, 0]
# temp = 0
# for i in y:
#     if i not in range(10):
#         y[temp] = 0
#     temp += 1
# x = data[:, 1:]
# x = x / 256
# x = np.reshape(x, (nums.SVHN_TRAIN + nums.CIFAR10_TRAIN, 28, 28))
# for i in range(6):
#     t = random.randint(0, nums.SVHN_TRAIN + nums.CIFAR10_TRAIN,)
#     img = x[t]
#     plt.imshow(img, cmap='gray')
#     print(y[t])
#     plt.show()
dict = {}
for i in y:
    if i not in dict:
        dict[i] = 1
    else:
        dict[i] += 1
print(dict)