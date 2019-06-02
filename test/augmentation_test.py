import resources.paths as paths
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import imgprocess.process as proc


fx = paths.PRODUCT_PATH + 'mnist_test.npy'
fy = paths.PRODUCT_PATH + 'mnist_test_label.npy'

x_train = np.load(fx)
y_train = np.load(fy)


print(np.shape(x_train))
x = x_train[11:12]
y = y_train[11:12]
print(y)
print(type(y))
config = {'rotation_range': 30, 'zoom_range': 0, 'horizontal_flip': False,
          'vertical_flip': False, 'loop_time': 4}
print('firstx:' + str(np.shape(x)))
print('firsty:' + str(np.shape(y)))


data_orig = {'x': x, 'y': y}
data_aug = proc.augment_for_gray(data_orig, config, (28, 28))
data_aug['x']/=256
print('augx: '+ str(np.shape(data_aug['x'])))
print('augy: '+ str(np.shape(data_aug['y'])))
print(data_aug['x'])
for i in data_aug['x']:
    plt.imshow(i)
    plt.show()
