
import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
import resources.paths as paths


def colorized(img):
    """convert gray img to RGB img"""
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    img_colorised = []
    for i in range(x):
        row = []
        for j in range(y):
            pixel = img[i][j]
            element = [pixel, pixel, pixel]
            row += [element]
        img_colorised += [row]
    return img_colorised


def cvt_color_and_size(im):
    """convert gray img to RGB img and change size to 32*32"""
    # 将数组转化为cv2.resize支持的uint8格式（原为int64)
    im = im.astype('uint8')
    # 将灰度图RGB化（算法采用R=G=B=Y)
    im_colored = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    # 尺寸转换
    im_colored_resized = cv2.resize(im_colored, (32, 32), interpolation=cv2.INTER_LINEAR)
    return im_colored_resized


def save_mnist(origin_file, dest_file, size):
    with open(origin_file, 'rb') as f1:
        buf1 = f1.read()
    image_index = 0
    im_cvt_list = []

    for i in range(size):
        try:
            image_index += struct.calcsize('>IIII')
            temp = struct.unpack_from('>784B', buf1, image_index)
            # '>784B'的意思就是用大端法读取784( 28*28 )个unsigned byte
            im = np.reshape(temp, (28, 28))
            im_cvt = cvt_color_and_size(im)
            im_cvt_list.append(im_cvt)
            image_index += struct.calcsize('>784B')
            print(i)
        except Exception:
            print('except')
            np.save(dest_file, im_cvt_list)
            return


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
        for e in elements[1: 785]:
            X.append(int(e))
    return X, y


def save_as_28281(inputf, out_data, out_label):
    (X, y) = read_file(inputf)
    X = np.array(X)
    size = int(np.shape(X)[0] / 784)
    y = np.array(y)
    X = np.reshape(X, (size, 28, 28))
    y = np.reshape(y, (np.shape(y)[0], 1))
    np.save(out_data, X)
    np.save(out_label, y)
    # print(np.shape(X))
    # print(np.shape(y))


def save_as_32323(inputf, out_data, out_label):
    (X, y) = read_file(inputf)
    X = np.array(X)
    y = np.array(y)
    size = (int)(np.shape(X)[0] / 784)
    X = np.reshape(X, (size, 28, 28))
    im_cvt = []
    for img in X:
        im_cvt.append(cvt_color_and_size(img))
    np.save(out_data, im_cvt)
    np.save(out_label, y)



#
# train_size = 60000
# test_size = 10000
#
# # 转换train data 为32323
# train_path = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/train-images-idx3-ubyte'
# train_save_path = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/MNIST/mnist_32323_train.npy'
# # save_mnist(train_path, train_save_path, train_size)
#
# # 转换test data 为32323
# test_path = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/t10k-images-idx3-ubyte'
# test_save_path = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/MNIST/mnist_32323_test.npy'
# # save_mnist(test_path, test_save_path, test_size)
#
# train_label = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/train-labels-idx1-ubyte'
# test_label = '/Users/kudengma/kudeng/Course/DeepLearning/dataset/t10k-labels-idx1-ubyte'


testf = paths.MNIST_PATH + 'mnist_test.csv'
trainf = paths.MNIST_PATH + 'mnist_train.csv'

#将MNIST数据以28*28*1格式存储
out_train = paths.PRODUCT_PATH + 'mnist_train.npy'
out_train_label = paths.PRODUCT_PATH + 'mnist_train_label.npy'
out_test = paths.PRODUCT_PATH + 'mnist_test.npy'
out_test_label = paths.PRODUCT_PATH + 'mnist_test_label.npy'
save_as_28281(trainf, out_train, out_train_label)
print('train 28281 success')
save_as_28281(testf, out_test, out_test_label)
print('test 28281 success')

# #将MNIST数据以32*32*3格式存储
# out_train_cvt = paths.PRODUCT_PATH + 'mnist_train_32323.npy'
# out_train_cvt_label = paths.PRODUCT_PATH + 'mnist_train_32323_label.npy'
# out_test_cvt = paths.PRODUCT_PATH + 'mnist_test_32323.npy'
# out_test_cvt_label = paths.PRODUCT_PATH + 'mnist_test_32323_label.npy'
# save_as_32323(trainf, out_train_cvt, out_train_cvt_label)
# print('train 32323 success')
# save_as_32323(testf, out_test_cvt, out_test_cvt_label)
# print('test 32323 success')




