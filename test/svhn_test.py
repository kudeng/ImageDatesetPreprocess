# read_file = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/google/svhn_28281'
# with open(read_file, "rb") as f1:
#     data = f1.read()
# print(data)
import  scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import resources.paths as paths

fx = paths.PRODUCT_PATH + 'svhn_train.npy'
x = np.load(fx)
print(np.shape(x))