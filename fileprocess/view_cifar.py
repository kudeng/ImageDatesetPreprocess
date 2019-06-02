import numpy as np
import resources.paths as paths
import cv2
import matplotlib.pyplot as plot

def unpickle(file):
    """cifar官方提供读取cifar数据的方式
       input: cifar下载的文件
       output: 以dict形式存取的包含data和label的数据
    """
    import pickle
    with open(file, 'rb') as fo:
        dictfile = pickle.load(fo, encoding='bytes')
    return dictfile


def reshape_and_combine():
    """
    将cifar提供的分块文件组合并输出
    :param origin_files: 分块文件的集合
    :param dest_data: data的存放路径
    :param dest_label: label的存放路径
    """
    total_num = 50000
    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    x = np.ndarray((total_num, 32, 32, 3))
    y = np.ndarray(total_num)
    for j in range(1, 6):
        dataName = paths.CIFAR_PATH + "data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
        Xtr = unpickle(dataName)
        print( "正在载入文件{}...".format(j))

        for i in range(0, 10000):
            img = np.reshape(Xtr[b'data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            x[(j-1)*10000+i] = img
        y[(j-1)*10000 : j*10000] = Xtr[b'labels']
        print("文件{}载入完毕".format(j))
    print("存储训练集...")
    dest_data = paths.PRODUCT_PATH + 'cifar10_train.npy'

    # y = np.reshape(y, (np.shape(y)[0], 1))
    x = np.reshape(x, (total_num, 32*32*3))
    train_data = np.ndarray((total_num, 32*32*3+1))
    train_data[:, 0] = y
    train_data[:, 1:] = x
    np.save(dest_data, train_data)

    print("训练集存储完成！")

    print("测试集载入中...")

    # 生成测试集图片
    testXtr = unpickle(paths.CIFAR_PATH + "test_batch")
    test_num = 10000
    x_test = np.ndarray((test_num, 32, 32, 3))
    y_test = np.ndarray(test_num)
    for i in range(0, 10000):
        img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        x_test[i] = img
    y_test = testXtr[b'labels']
    print("测试集载入完成.")
    test_data_path = paths.PRODUCT_PATH + 'cifar10_test.npy'
    test_data = np.ndarray((test_num, 32*32*3+1))
    x_test = np.reshape(x_test, (test_num, 32*32*3))
    test_data[:, 0] = y_test
    test_data[:, 1:] = x_test
    print("存储测试集...")
    np.save(test_data_path, test_data)

    print("测试集存储完成！")




def resize_and_colorise(img):
    """
    将32*32彩色图转换为28*28灰度图
    :param img:
    :return:
    """
    img_colorised = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_colorised_resized = cv2.resize(img_colorised, (28, 28))
    return img_colorised_resized


def view_cifar10():
    """
    生成cifar10训练集和测试集 32*32*3
    :return:
    """
    # 生成cifar10训练集 32*32*3
    path = paths.CIFAR_PATH
    file1 = path + 'data_batch_1'
    file2 = path + 'data_batch_2'
    file3 = path + 'data_batch_3'
    file4 = path + 'data_batch_4'
    file5 = path + 'data_batch_5'
    train_files = [file1, file2, file3, file4, file5]
    dest_data = paths.PRODUCT_PATH + 'cifar10_train_32323.npy'
    dest_label = paths.PRODUCT_PATH + 'cifar10_train_32323_label.npy'
    (X, y) = reshape_and_combine(train_files)
    np.save(dest_data, X)
    np.save(dest_label, y)

    # 生成cifar10测试集 32*32*3
    test_file = paths.CIFAR_PATH + 'test_batch'
    test_files = [test_file]
    test_data = paths.PRODUCT_PATH + 'cifar10_test_32323.npy'
    test_label = paths.PRODUCT_PATH + 'cifar10_test_32323_label.npy'
    (X_test, y_test) = reshape_and_combine(test_files)
    np.save(test_data, X_test)
    np.save(test_label, y_test)


# def test():
#     """
#     生成cifar10训练集和测试集 32*32*3
#     :return:
#     """
#     # 生成cifar10训练集 32*32*3
#     path = paths.CIFAR_PATH
#     file1 = path + 'data_batch_1'
#     train_files = [file1]
#     reshape_and_combine()
#     print(np.shape(X[1]))
#     for i in range(3):
#         plot.imshow(X[i])
#         plot.show()


reshape_and_combine()
