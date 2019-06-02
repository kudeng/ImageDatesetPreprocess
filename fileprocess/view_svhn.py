import scipy.io as scio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import resources.paths as paths


path_train = paths.SVHN_PATH + 'train_32x32.mat'
data = scio.loadmat(path_train)
print(data.keys())
print(data.get('__header__'))
print(data.get('__version__'))
print(data.get('__globals__'))

def svhn_to_mnist(X_cvt, origin_file):
    data = scio.loadmat(origin_file)
    X = data.get('X').transpose(3, 0 ,1 ,2)
    # # 测试尺寸、通道转换， 状态：成功，可用函数：opencv::resize, opencv::cvtColor
    # img = X.T[0].T
    # img_resized = cv2.resize(img, (28, 28))
    # img_resized_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # print(np.shape(img_resized_gray))
    # print(img_resized_gray)
    # plt.imshow(img_resized_gray, cmap = 'gray')
    # plt.show()
    #转换所有图片
    for img in X.T:
        img = img.T
        img_resized = cv2.resize(img, (28, 28))
        img_resized_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        X_cvt += [img_resized_gray]
    return X_cvt


def trans_train():
    X_cvt0 = []
    ##加载并变换train
    path_train = paths.SVHN_PATH + 'train_32x32.mat'
    X_cvt1 = svhn_to_mnist(X_cvt0, path_train)
    y_train = scio.loadmat(path_train).get('y')
    ##加载并变换extra  附加到上述数据
    path_extra = paths.SVHN_PATH + 'extra_32x32.mat'
    X_cvt2 = svhn_to_mnist(X_cvt1, path_extra)
    y_extra = scio.loadmat(path_extra).get('y')
    train_data_path = paths.PRODUCT_PATH + 'svhn_train_28281.npy'
    train_label_path = paths.PRODUCT_PATH + 'svhn_train_28281_label.npy'
    np.save(train_data_path, X_cvt2)
    y_cvt = np.append(y_train, y_extra)
    np.save(train_label_path, y_cvt)


def trans_test():
    X_cvt0 = []
    ##加载并变换test
    path_test = paths.SVHN_PATH + 'test_32x32.mat'
    X_cvt1 = svhn_to_mnist(X_cvt0, path_test)
    y_test = scio.loadmat(path_test).get('y')
    test_data_path = paths.PRODUCT_PATH + 'svhn_test_28281.npy'
    test_label_path = paths.PRODUCT_PATH + 'svhn_test_28281_label.npy'
    np.save(test_data_path, X_cvt1)
    np.save(test_label_path, y_test)


def view_svhn_28281():
    """
    将svhn数据度读取并以npy格式保存在本地
    :return:
    """
    #填写路径
    train_data = paths.SVHN_PATH + 'train_32x32.mat'
    extra_data = paths.SVHN_PATH + 'extra_32x32.mat'
    test_data = paths.SVHN_PATH + 'test_32x32.mat'
    train_out = paths.PRODUCT_PATH + 'svhn_train.npy'
    train_out_label = paths.PRODUCT_PATH + 'svhn_train_label.npy'
    test_out = paths.PRODUCT_PATH + 'svhn_test.npy'
    test_out_label = paths.PRODUCT_PATH + 'svhn_test_label.npy'

    #载入数据
    print("正在载入svhn训练集数据...")
    data1 = scio.loadmat(train_data)
    X1 = data1.get('X').transpose(3, 0, 1, 2)
    y1 = data1.get('y')
    del data1

    data2 = scio.loadmat(extra_data)
    X2 = data2.get('X').transpose(3, 0, 1, 2).astype('uint8')
    y2 = data2.get('y')
    del data2

    X = np.append(X1, X2)
    x_num = int(np.shape(X)[0] / (32 * 32 * 3))
    X = np.reshape(X, (x_num, 32, 32, 3))

    y = np.append(y1, y2)
    y_num = np.shape(y)[0]
    y = np.reshape(y, (y_num, 1))

    print("正在保存svhn训练集数据...")
    np.save(train_out, X)
    np.save(train_out_label, y)
    print("保存成功！")

    print("正在载入svhn测试集数据...")
    data_test = scio.loadmat(test_data)
    X_test = data_test.get('X').transpose(3, 0 ,1 ,2)
    y_test = data_test.get('y')
    y_test_num = np.shape(y_test)[0]
    y_test = np.reshape(y_test, (y_test_num, 1))

    print("正在保存测试集数据...")
    np.save(test_out, X_test)
    np.save(test_out_label, y_test)
    print("保存成功！")
    print("存储svhn完毕！")


def test():
    train_data = paths.SVHN_PATH + 'train_32x32.mat'
    data = scio.loadmat(train_data)
    X = data['X'].transpose(3, 0 ,1 ,2)
    for i in range(3):
        plt.imshow(X[i])
        plt.show()


# img_resized = cv2.resize(img, (28, 28))
# img_resized_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
# print(np.shape(img_resized_gray))
# print(img_resized_gray)
# plt.imshow(img_resized_gray, cmap = 'gray')
# plt.show()

# view_svhn_28281()