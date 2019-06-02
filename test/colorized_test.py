import resources.paths as paths
import numpy as np
import matplotlib.pyplot as plt
import imgprocess.process as proc


fx = paths.PRODUCT_PATH + 'mnist_test.npy'
fy = paths.PRODUCT_PATH + 'mnist_test_label.npy'

# fx = paths.PRODUCT_PATH + 'svhn_train_32323.npy'
# fy = paths.PRODUCT_PATH + 'svhn_train_32323_label.npy'

x_train = np.load(fx)
y_train = np.load(fy)
y_train = np.reshape(y_train, (np.shape(y_train)[0], 1))

x_train = proc.resize_and_colorise(x_train, (32, 32))
print(np.shape(x_train))


x = x_train[91:92]
y = y_train[91:92]

# for img in x:
#     plt.imshow(img)
#     plt.show()

config = {'rotation_range': 30, 'zoom_range': 0, 'horizontal_flip': False,
          'vertical_flip': False, 'loop_time': 2}
print('firstx:' + str(np.shape(x)))
print('firsty:' + str(np.shape(y)))


data_orig = {'x': x, 'y': y}
data_aug = proc.augment(data_orig, config)
print('augx: ' + str(np.shape(data_aug['x'])))
print('augy: ' + str(np.shape(data_aug['y'])))
print(data_aug['y'])
for i in data_aug['x']:
    plt.imshow(i)
    plt.show()
