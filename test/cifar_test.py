import numpy as np
import struct
import cv2
import resources.paths as paths
import matplotlib.pyplot as plt

fx = paths.PRODUCT_PATH + 'cifar10_train.npy'
fy = paths.PRODUCT_PATH + 'cifar10_train_label.npy'

x = np.load(fx).astype('uint8')
x /= 256
y = np.load(fy)

count = 0
for img in x[0:5]:
    plt.imshow(img)
    plt.show()
    print(y[count])
    count += 1