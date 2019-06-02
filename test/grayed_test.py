import resources.paths as paths
import numpy as np
import matplotlib.pyplot as plt
import imgprocess.process as proc
import cv2


fx = paths.PRODUCT_PATH + 'cifar10_test.npy'
fy = paths.PRODUCT_PATH + 'cifar10_test_label.npy'
x_train = np.load(fx)
y_train = np.load(fy)

x = x_train[0:2]
y = y_train[0:2]


config = {'rotation_range': 30, 'shear_range': 0, 'zoom_range': 0, 'horizontal_flip': False,
          'vertical_flip': False, 'loop_time': 2}

data_orig = {'x': x, 'y': y}
data_aug = proc.augment_for_gray(data_orig, config, (28, 28))

print(np.shape(data_aug['x']))
count = 0
for img in data_aug['x']:
    plt.imshow(img, cmap="gray")
    plt.show()
    print(data_aug['y'][count])
    count += 1
print('augx: '+ str(np.shape(data_aug['x'])))
print('augy: '+ str(np.shape(data_aug['y'])))
# for img in data_aug['x']:
#     plt.imshow(img, cmap='gray')
#     plt.show()
# for img in data_aug['x']:
#     plt.imshow(img)
#     plt.show()