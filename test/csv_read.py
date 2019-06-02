from resources import paths
import  csv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

file = paths.OUT_PATH + "20190506_193351_test.csv"
# file = paths.PRODUCT_PATH + '20190506_191606_test.csv'
# df = pd.read_csv(file, header=None, dtype='uint8')
with open(file, 'r') as f1:
    for i in range(10):
        line = f1.readline()
        print(line)
# batch_size = 32
# num = 100
# sub_df = df[num*batch_size:(num+1)*batch_size].values
# print(sub_df)
# y = sub_df[:, 0]
# print(y)
# print(len(df))
# batch_size = 32
# num = 0
# while True:
#     #获取x y
#     sub_df = df[num*batch_size:(num+1)*batch_size]
#     if (len(sub_df)==0):
#         break
#     print("num %d" % num)
#     actual_size = len(sub_df)
#     ratio = 60000 / batch_size
#     if len == 0:
#         break
#     data = sub_df.values
#     y = data[:, 0]
#     print(y)
#     x = data[:, 1:]
#     print(x)
#     num += 1
#     print(num / ratio)

# data = df[10:10]
# print(data.size)
# print(len(data))
# print(df[10:20][0])

    # spamreader = csv.reader(f1, delimiter=' ', quotechar='|')
    # for line in spamreader:
    #     data = line[0].split(',')
    #     y = data[0]
    #     x = data[1:]
    #     print('y:')
    #     print(y)
    #     print('x')
    #     print(x)
    #     print(type(y))
    #     print(type(x[3]))

    # line = f1.readline()
    # elements = line.split(',')
    # print(elements)
    # x = []
    # for e in elements[1:]:
    #     x.append(int(e))
    # x = np.array(x)
    # x = np.reshape(x, (32, 32, 3))
    # print(np.shape(x))
    # print(x)
    # plt.imshow(x)
    # plt.show()