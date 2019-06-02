import numpy as np
import struct
import cv2
import resources.paths as paths
import matplotlib.pyplot as plt


def cvtColorAndSize(im):
    # 将数组转化为cv2.resize支持的uint8格式（原为int64)
    im = im.astype('uint8')
    # 手工将灰度图RGB化（算法采用R=G=B=Y)
    im_colored = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    # 尺寸转换
    im_colored_resized = cv2.resize(im_colored, (512, 512), interpolation=cv2.INTER_CUBIC)
    return im_colored_resized


train = paths.PRODUCT_PATH + 'mnist_train.npy'
x = np.load(train)
for img in x[0:5]:
    plt.imshow(img, cmap='gray')
    plt.show()
