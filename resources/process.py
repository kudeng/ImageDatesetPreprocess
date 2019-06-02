import cv2
import keras.preprocessing.image as img
import numpy as np


def resize_and_gray(imgs, size):
    """
    将彩色图转换为所需尺寸灰度图
    :param imgs:
    :param size:
    :return:
    """
    num = np.shape(imgs)[0]
    (width, height) = size
    imgs_processed = np.ndarray((num, width, height))
    imgs = imgs.astype('uint8')
    count = 0
    for img in imgs:
        img_grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_grayed_resized = cv2.resize(img_grayed, size)
        imgs_processed[count] = img_grayed_resized
        count += 1
    return imgs_processed


def resize_and_colorise(imgs, size):
    """
    将灰度图转换为所需尺寸彩色图
    :param imgs:
    :param size:
    :return:
    """
    num = np.shape(imgs)[0]
    (width, height) = size
    imgs_processed = np.ndarray((num, width, height, 3))
    imgs = imgs.astype('uint8')
    count = 0
    for img in imgs:
        img_colorised = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_colorised_resized = cv2.resize(img_colorised, size, interpolation=cv2.INTER_CUBIC)
        imgs_processed[count] = img_colorised_resized
        count += 1
    return imgs_processed


def resize(imgs, size):
    """
    将彩色图按需更改尺寸
    :param img:
    :param size:
    :return:
    """
    num = np.shape(imgs)[0]
    (width, height) = size
    imgs_processed = np.ndarray((num, width, height, 3))
    imgs = imgs.astype('uint8')
    count = 0
    for img in imgs:
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        imgs_processed[count] = img_resized
        count += 1
    return imgs_processed


def resize_for_gray(imgs, size):
    """
    将灰度图/彩色图按需更改尺寸
    :param img:
    :param size:
    :return:
    """
    num = np.shape(imgs)[0]
    (width, height) = size
    imgs_processed = np.ndarray((num, width, height))
    # imgs = imgs.astype('uint8')
    count = 0
    for img in imgs:
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        imgs_processed[count] = img_resized
        count += 1
    return imgs_processed


def colorize(imgs):
    imgs = imgs.astype('uint8')
    (num,x,y) = np.shape(imgs)
    imgs_processed = np.ndarray((num, x, y, 3))
    count = 0
    for img in imgs:
        img_colorised = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        imgs_processed[count] = img_colorised
        count += 1
    return imgs_processed


# def grayize(imgs):
#     (num,x,y,z) = np.shape(imgs)
#     imgs_processed = np.ndarray((num, x, y))
#     count = 0
#     for img in imgs:
#         img_colorised = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         imgs_processed[count] = img_colorised
#         count += 1
#     return imgs_processed


def augment(x, y, config):
    """
    对彩色图按需进行数据增强
    :param x:
    :param config:
    :return:
    """
    # print("数据增强启动，正在载入数据与配置...")
    datagen = img.ImageDataGenerator(rotation_range=config['rotation_range'],
                                     zoom_range=config['zoom_range'],
                                     horizontal_flip=config['horizontal_flip'],
                                     vertical_flip=config['vertical_flip'],
                                     )
    it = datagen.flow(x, y=y, batch_size=80000, shuffle=True,
                      sample_weight=None, seed=None, save_to_dir=None,
                      save_prefix=None, save_format='png', subset=None)
    # print("装载完成，启动超级变换形态...")
    return it.next()



def grayize(x):
    """
    将图像灰度化
    :param x:
    :return:
    """
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])