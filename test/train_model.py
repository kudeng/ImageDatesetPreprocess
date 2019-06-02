# -*- coding: utf-8 -*-
import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from resources import paths
from resources import nums
import math
batch_size = 128
num_classes = 10
epochs = 10


num = nums.CIFAR10_TRAIN
test_num = nums.CIFAR10_TEST

path= paths.OUT_PATh + 'cifar_rotate.npy'
path2 = paths.PRODUCT_PATH + 'cifar10_test.npy'
f1 = np.load(path)
f2 = np.load(path2)
x_train, y_train = f1[:, 1:], f1[:, 0]
x_test, y_test = f2[:, 1:], f2[:, 0]
y_train = y_train - 10
# f1.close()
# f2.close()

x_train = x_train.reshape(num, 3072).astype('float32')
x_test = x_test.reshape(test_num, 3072).astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 全连接模型
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(3072,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#损失函数使用交叉熵
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
#模型估计
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Total loss on Test Set:', score[0])
print('Accuracy of Testing Set:', score[1])

#预测
result = model.predict_classes(x_test)
correct_indices = np.nonzero(result == y_test)[0]
incorrect_indices = np.nonzero(result != y_test)[0]
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(result[correct], y_test[correct]))

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(result[incorrect], y_test[incorrect]))
    plt.show()