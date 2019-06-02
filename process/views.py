import csv
import threading
import time
from resources import nums
from concurrent.futures import thread
import operator

import cv2
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.db import connection
from django.contrib import auth
from django.contrib.auth.models import User
from django.http import FileResponse
from resources import paths
from resources import process
import random
from matplotlib import pyplot as plt

import pandas as pd

# Create your views here.

def unpickle(file):
    """cifar官方提供读取cifar数据的方式
       input: cifar下载的文件
       output: 以dict形式存取的包含data和label的数据
    """
    import pickle
    with open(file, 'rb') as fo:
        dictfile = pickle.load(fo, encoding='bytes')
    return dictfile


def index(request):
    return render(request, "index.html")


def signup(request):
    return render(request, "sign-up.html")


def forgot(request):
    return render(request, "forgot.html")


def helpme(request):
    file = paths.CIFAR100_PATH + 'meta'
    meta = unpickle(file)
    fine_labels = meta.get(b'fine_label_names')
    i = 0
    for fine_label in fine_labels:
        label = fine_label.decode('utf-8')
        fine_labels[i] = str(100 + i) + ": " + label
        i += 1
    context = {
        'fine_labels': fine_labels,
    }
    return render(request, "help.html", context=context)


def profile(request):
    return render(request, "profile.html")


def table(request):
    if request.method == 'GET':
        return render(request, "table.html")

    if request.method == 'POST':
        post = request.POST

        user_id = request.user.id
        cursor = connection.cursor()
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sql = "INSERT INTO task(create_time, state, content, user_id) VALUES ('%s', %d, '%s', %d)" % (str(now), 2, 'unfinished', user_id)
        cursor.execute(sql)
        t1 = threading.Thread(target=producing, args=(request, now))
        t1.start()
        return render(request, "table.html")


def task(request):
    if request.method == 'GET':
        cursor = connection.cursor()
        active_user = request.user
        user_id = active_user.id
        sql = "SELECT id, create_time, content, state, train_path, test_path FROM task WHERE user_id = %d" % \
              user_id
        cursor.execute(sql)
        tasks = cursor.fetchall()
        context = {
            'transactions': tasks,
            'user': active_user
        }
        return render(request, "task.html", context=context)
    if request.method == 'POST':
        print("task")
        post = request.POST
        print(request.body)
        task_id = int(post.get("preview_task"))
        cursor = connection.cursor()
        sql = "SELECT content, train_path FROM task WHERE id = %d" % \
              task_id
        cursor.execute(sql)
        tasks = cursor.fetchall()
        content = tasks[0][0]
        file = tasks[0][1]
        rows = content.split('\n')
        img_conf = rows[0].split('*')
        width, height, channel = int(img_conf[0]), int(img_conf[1]), int(img_conf[2])
        data = np.load(paths.OUT_PATh + file)
        total = np.shape(data)[0]
        img_path = '/Users/kudengma/PycharmProjects/DLPreProcWeb/templates/static/homepage/preview.png'
        x = data[:, 1:]
        y = data[:, 0]

        fig = plt.figure()

        if channel == 3:
            x = np.reshape(x, (total, width, height, channel))
            for i in range(9):
                randn = random.randint(0, total)
                img = x[randn]
                img /= 256
                ax = fig.add_subplot(331 + i)
                ax.title.set_text(str(y[randn]))
                plt.imshow(img)

        if channel == 1:
            x = np.reshape(x, (total, width, height))
            for i in range(9):
                randn = random.randint(0, total)
                img = x[randn]
                img /= 256
                ax = fig.add_subplot(331 + i)
                ax.title.set_text(str(y[randn]))
                plt.imshow(img, cmap='gray')

        plt.savefig(img_path)
        print("save success")
        return HttpResponseRedirect("preview.html")


def djlogin(request):
    if request.method == 'GET':
        return render(request, 'index.html')

    if request.method == 'POST':

        name = request.POST.get('username')
        password = request.POST.get('password')
        # 验证用户名和密码，验证通过的话，返回user对象
        user = auth.authenticate(username=name, password=password)
        print("authenticated")
        if user:
            # 验证成功 登陆
            auth.login(request, user)
            print("login")
            return HttpResponseRedirect("table.html")
        else:
            return render(request, 'index.html')


def djregist(request):
    if request.method == 'GET':
        return render(request, 'sign-up.html')
    if request.method == 'POST':
        name = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        print(request)
        print("email:")
        print(request.POST.get("email"))
        # 此处的User 是 django 自带的model
        User.objects.create_user(username=name, password=password, email=email)
        user = auth.authenticate(username=name, password=password)
        auth.login(request, user)
        return HttpResponseRedirect("table.html")


def djlogout(request):
    if request.method == 'GET':
        auth.logout(request)
        return HttpResponseRedirect("index.html")


def download(request, file_name):
    abs_path = paths.OUT_PATh + file_name
    file = open(abs_path, 'rb')
    response = FileResponse(file)
    response['Content-Type']='application/octet-stream'
    response['Content-Disposition'] = '''attachment;filename=''' + '''"''' + file_name + '''"'''
    return response


def preview(request):
        return render(request, "preview.html")


def producing(request, creat_time):
    post = request.POST
    print(request.body)
    datasets = []
    mnist_content = ""
    svhn_content = ""
    cifar10_content = ""
    cifar100_content = ""
    trainset_num = 0
    testset_num = 0
    if post.get("mnist") is not None:
        datasets.append("mnist")
        trainset_num += nums.MNIST_TRAIN
        testset_num += nums.MNIST_TEST
        mnist_content += 'mnist'
    if post.get("svhn") is not None:
        datasets.append("svhn")
        trainset_num += nums.SVHN_TRAIN
        testset_num += nums.SVHN_TEST
        svhn_content += 'svhn'
    if post.get("cifar10") is not None:
        datasets.append("cifar10")
        trainset_num += nums.CIFAR10_TRAIN
        testset_num += nums.CIFAR10_TEST
        cifar10_content += 'cifar10'
    if post.get("cifar100") is not None:
        datasets.append("cifar100")
        trainset_num += nums.CIFAR100_TRAIN
        testset_num += nums.CIFAR100_TEST
        cifar100_content += 'cifar100'
    print(datasets)
    isaugment = False
    if post.get("img_augment") == 'y':
        isaugment = True
    width = int(post.get("img_width"))
    height = int(post.get("img_height"))
    channel = int(post.get("img_channel"))
    header = "%d*%d*%d\n" % (width, height, channel)
    trainset = np.ndarray((trainset_num, width*height*channel+1))
    testset = np.ndarray((testset_num, width*height*channel+1))

    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    train_outf = paths.OUT_PATh + str(now) + '_train.npy'
    train_outf_name = str(now) + '_train.npy'
    test_outf = paths.OUT_PATh + str(now) + '_test.npy'
    test_outf_name = str(now) + '_test.npy'

    train_index = 0
    test_index = 0

    for dataset in datasets:
        #对mnist进行处理
        if dataset == 'mnist':
            # 载入当前数据库数据
            train_data = np.load(paths.PRODUCT_PATH + dataset + "_train.npy")
            y = train_data[:, 0]
            x = train_data[:, 1:]
            del train_data
            print('正在处理MNIST训练集...')
            x = np.reshape(x, (nums.MNIST_TRAIN, 28, 28))
            if width != 28 or height != 28:
                x = process.resize_for_gray(x, (width, height))
            #获取数据增强参数
            if isaugment:
                if post.get("rotation_" + dataset) is not None:
                    rotation = int(post.get("rotation_" + dataset))
                    mnist_content = mnist_content + '_rotate%d' % rotation
                else:
                    rotation = 0
                if post.get("zoom_" + dataset) is not None:
                    zoom = float(post.get("zoom_" + dataset))
                    mnist_content = mnist_content + '_zoom%f' % zoom
                else:
                    zoom = 0
                horizontal = False
                vertical = False
                if post.get("horizontal_" + dataset) is not None:
                    horizontal = True
                    mnist_content = mnist_content + '_horizontaled'
                if post.get("vertical_" + dataset) is not None:
                    vertical = True
                    mnist_content = mnist_content + '_verticaled'
                config = {
                    'rotation_range': rotation,
                    'zoom_range': zoom,
                    'horizontal_flip': horizontal,
                    'vertical_flip': vertical,
                }
                x = process.colorize(x)
                print('color:' + str(np.shape(x)))
                (x, y) = process.augment(x, y, config)
                print('augment:' + str(np.shape(x)))
                if channel == 1:
                    x = process.grayize(x)
                    print('gray:' + str(np.shape(x)))
            elif channel == 3:
                x = process.colorize(x)
            x = np.reshape(x, (nums.MNIST_TRAIN, width*height*channel))
            trainset[train_index: train_index + nums.MNIST_TRAIN, 0] = y
            trainset[train_index: train_index + nums.MNIST_TRAIN, 1:] = x
            train_index += nums.MNIST_TRAIN
            mnist_content += '\n'
            print("MNIST训练集处理完成！")

            print("正在处理MNIST测试集...")
            test_data = np.load(paths.PRODUCT_PATH + dataset + "_test.npy")
            y = test_data[:, 0]
            x = test_data[:, 1:]
            del test_data
            print('正在处理MNIST测试集...')
            x = np.reshape(x, (nums.MNIST_TEST, 28, 28))
            if width != 28 or height != 28:
                x = process.resize_for_gray(x, (width, height))
            # # 获取数据增强参数
            # if isaugment:
            #     if post.get("rotation_" + dataset) is not None:
            #         rotation = int(post.get("rotation_" + dataset))
            #
            #     else:
            #         rotation = 0
            #     if post.get("zoom_" + dataset) is not None:
            #         zoom = float(post.get("zoom_" + dataset))
            #         mnist_content = mnist_content + '_zoom%f' % zoom
            #     else:
            #         zoom = 0
            #     horizontal = False
            #     vertical = False
            #     if post.get("horizontal_" + dataset) is not None:
            #         horizontal = True
            #         mnist_content = mnist_content + '_horizontaled'
            #     if post.get("vertical_" + dataset) is not None:
            #         vertical = True
            #         mnist_content = mnist_content + '_verticaled'
            #     config = {
            #         'rotation_range': rotation,
            #         'zoom_range': zoom,
            #         'horizontal_flip': horizontal,
            #         'vertical_flip': vertical,
            #     }
            #     x = process.colorize(x)
            #     print('color:' + str(np.shape(x)))
            #     (x, y) = process.augment(x, y, config)
            #     print('augment:' + str(np.shape(x)))
            #     if channel == 1:
            #         x = process.grayize(x)
            #         print('gray:' + str(np.shape(x)))
            # elif channel == 3:
            #     x = process.colorize(x)
            # x = np.reshape(x, (nums.MNIST_TEST, width * height * channel))
            if channel == 3:
                x = process.colorize(x)
            x = np.reshape(x, (nums.MNIST_TEST, width*height*channel))
            testset[test_index: test_index + nums.MNIST_TEST, 0] = y
            testset[test_index: test_index + nums.MNIST_TEST, 1:] = x
            test_index += nums.MNIST_TEST
            mnist_content += '\n'
            print("MNIST测试集处理完成！")

        if dataset == 'svhn':
            # 载入当前数据库数据
            train_data = np.load(paths.PRODUCT_PATH + dataset + "_train.npy")
            y = train_data[:, 0]
            temp = 0
            for i in y:
                if int(i)==10:
                    y[temp] = 0
                temp += 1
            x = train_data[:, 1:]
            del train_data
            print('正在处理SVHN训练集...')
            x = np.reshape(x, (nums.SVHN_TRAIN, 32, 32, 3))
            if width != 32 or height != 32:
                x = process.resize(x, (width, height))
            #获取数据增强参数
            if isaugment:
                if post.get("rotation_" + dataset) is not None:
                    rotation = int(post.get("rotation_" + dataset))
                    svhn_content += '_rotate%d' % rotation
                else:
                    rotation = 0
                if post.get("zoom_" + dataset) is not None:
                    zoom = float(post.get("zoom_" + dataset))
                    svhn_content += '_zoom%f' % zoom
                else:
                    zoom = 0
                horizontal = False
                vertical = False
                if post.get("horizontal_" + dataset) is not None:
                    horizontal = True
                    svhn_content += '_horizontaled'
                if post.get("vertical_" + dataset) is not None:
                    vertical = True
                    svhn_content += '_verticaled'
                config = {
                    'rotation_range': rotation,
                    'zoom_range': zoom,
                    'horizontal_flip': horizontal,
                    'vertical_flip': vertical,
                }
                print('color:' + str(np.shape(x)))
                (x, y) = process.augment(x, y, config)
                print('augment:' + str(np.shape(x)))
                if channel == 1:
                    x = process.grayize(x)
                    print('gray:' + str(np.shape(x)))
            elif channel == 1:
                x = process.grayize(x)
            x = np.reshape(x, (nums.SVHN_TRAIN, width*height*channel))
            trainset[train_index: train_index + nums.SVHN_TRAIN, 0] = y
            trainset[train_index: train_index + nums.SVHN_TRAIN, 1:] = x
            train_index += nums.SVHN_TRAIN
            svhn_content += '\n'
            print("SVHN训练集处理完成！")

            # 载入当前数据库数据
            print('正在处理SVHN测试集...')
            test_data = np.load(paths.PRODUCT_PATH + dataset + "_test.npy")
            y = test_data[:, 0]
            temp = 0
            for i in y:
                if int(i) == 10:
                    y[temp] = 0
                temp += 1
            x = test_data[:, 1:]
            x = np.reshape(x, (nums.SVHN_TEST, 32, 32, 3))
            if width != 32 or height != 32:
                x = process.resize(x, (width, height))
            # # 获取数据增强参数
            # if isaugment:
            #     if post.get("rotation_" + dataset) is not None:
            #         rotation = int(post.get("rotation_" + dataset))
            #         svhn_content += '_rotate%d' % rotation
            #     else:
            #         rotation = 0
            #     if post.get("zoom_" + dataset) is not None:
            #         zoom = float(post.get("zoom_" + dataset))
            #         svhn_content += '_zoom%f' % zoom
            #     else:
            #         zoom = 0
            #     horizontal = False
            #     vertical = False
            #     if post.get("horizontal_" + dataset) is not None:
            #         horizontal = True
            #         svhn_content += '_horizontaled'
            #     if post.get("vertical_" + dataset) is not None:
            #         vertical = True
            #         svhn_content += '_verticaled'
            #     config = {
            #         'rotation_range': rotation,
            #         'zoom_range': zoom,
            #         'horizontal_flip': horizontal,
            #         'vertical_flip': vertical,
            #     }
            #     print('color:' + str(np.shape(x)))
            #     (x, y) = process.augment(x, y, config)
            #     print('augment:' + str(np.shape(x)))
            #     if channel == 1:
            #         x = process.grayize(x)
            #         print('gray:' + str(np.shape(x)))
            # elif channel == 1:
            #     x = process.grayize(x)
            if channel == 1:
                x = process.grayize(x)
            x = np.reshape(x, (nums.SVHN_TEST, width*height*channel))
            testset[test_index: test_index + nums.SVHN_TEST, 0] = y
            testset[test_index: test_index + nums.SVHN_TEST, 1:] = x
            test_index += nums.SVHN_TEST
            svhn_content += '\n'
            print("SVHN测试集处理完成！")

        if dataset == 'cifar10':
            # 载入当前数据库数据
            train_data = np.load(paths.PRODUCT_PATH + dataset + "_train.npy")
            y = train_data[:, 0] + 10
            x = train_data[:, 1:]
            del train_data
            print('正在处理CIFAR10训练集...')
            x = np.reshape(x, (nums.CIFAR10_TRAIN, 32, 32, 3))
            if width != 32 or height != 32:
                x = process.resize(x, (width, height))
            # 获取数据增强参数
            if isaugment:
                if post.get("rotation_" + dataset) is not None:
                    rotation = int(post.get("rotation_" + dataset))
                    cifar10_content += '_rotate%d' % rotation
                else:
                    rotation = 0
                if post.get("zoom_" + dataset) is not None:
                    zoom = float(post.get("zoom_" + dataset))
                    cifar10_content += '_zoom%f' % zoom
                else:
                    zoom = 0
                horizontal = False
                vertical = False
                if post.get("horizontal_" + dataset) is not None:
                    horizontal = True
                    cifar10_content += '_horizontaled'
                if post.get("vertical_" + dataset) is not None:
                    vertical = True
                    cifar10_content += '_verticaled'
                config = {
                    'rotation_range': rotation,
                    'zoom_range': zoom,
                    'horizontal_flip': horizontal,
                    'vertical_flip': vertical,
                }
                print('color:' + str(np.shape(x)))
                (x, y) = process.augment(x, y, config)
                print('augment:' + str(np.shape(x)))
                if channel == 1:
                    x = process.grayize(x)
                    print('gray:' + str(np.shape(x)))
            elif channel == 1:
                x = process.grayize(x)
            x = np.reshape(x, (nums.CIFAR10_TRAIN, width * height * channel))
            trainset[train_index: train_index + nums.CIFAR10_TRAIN, 0] = y
            trainset[train_index: train_index + nums.CIFAR10_TRAIN, 1:] = x
            train_index += nums.CIFAR10_TRAIN
            cifar10_content += '\n'
            print("CIFAR10训练集处理完成！")

            # 载入当前数据库数据
            print('正在处理CIFAR10测试集...')
            test_data = np.load(paths.PRODUCT_PATH + dataset + "_test.npy")
            test_data[:, 0] += 10
            x = test_data[:, 1:]
            x = np.reshape(x, (nums.CIFAR10_TEST, 32, 32, 3))
            if width != 32 or height != 32:
                x = process.resize(x, (width, height))
            # # 获取数据增强参数
            # if isaugment:
            #     if post.get("rotation_" + dataset) is not None:
            #         rotation = int(post.get("rotation_" + dataset))
            #         cifar10_content += '_rotate%d' % rotation
            #     else:
            #         rotation = 0
            #     if post.get("zoom_" + dataset) is not None:
            #         zoom = float(post.get("zoom_" + dataset))
            #         cifar10_content += '_zoom%f' % zoom
            #     else:
            #         zoom = 0
            #     horizontal = False
            #     vertical = False
            #     if post.get("horizontal_" + dataset) is not None:
            #         horizontal = True
            #         cifar10_content += '_horizontaled'
            #     if post.get("vertical_" + dataset) is not None:
            #         vertical = True
            #         cifar10_content += '_verticaled'
            #     config = {
            #         'rotation_range': rotation,
            #         'zoom_range': zoom,
            #         'horizontal_flip': horizontal,
            #         'vertical_flip': vertical,
            #     }
            #     print('color:' + str(np.shape(x)))
            #     (x, y) = process.augment(x, y, config)
            #     print('augment:' + str(np.shape(x)))
            #     if channel == 1:
            #         x = process.grayize(x)
            #         print('gray:' + str(np.shape(x)))
            # elif channel == 1:
            #     x = process.grayize(x)
            if channel == 1:
                x = process.grayize(x)
            x = np.reshape(x, (nums.CIFAR10_TEST, width*height*channel))
            testset[test_index: test_index + nums.CIFAR10_TEST, 0] = test_data[:, 0]
            testset[test_index: test_index + nums.CIFAR10_TEST, 1:] = x
            test_index += nums.CIFAR10_TEST
            print("CIFAR10测试集处理完成！")

        if dataset == 'cifar100':
            # 载入当前数据库数据
            train_data = np.load(paths.PRODUCT_PATH + dataset + "_train.npy")
            y = train_data[:, 0] + 100
            x = train_data[:, 1:]
            del train_data
            print('正在处理CIFAR100训练集...')
            x = np.reshape(x, (nums.CIFAR100_TRAIN, 32, 32, 3))
            if width != 32 or height != 32:
                x = process.resize(x, (width, height))
            # 获取数据增强参数
            if isaugment:
                if post.get("rotation_" + dataset) is not None:
                    rotation = int(post.get("rotation_" + dataset))
                    cifar100_content += '_rotate%d' % rotation
                else:
                    rotation = 0
                if post.get("zoom_" + dataset) is not None:
                    zoom = float(post.get("zoom_" + dataset))
                    cifar100_content += '_zoom%f' % zoom
                else:
                    zoom = 0
                horizontal = False
                vertical = False
                if post.get("horizontal_" + dataset) is not None:
                    horizontal = True
                    cifar100_content += '_horizontaled'
                if post.get("vertical_" + dataset) is not None:
                    vertical = True
                    cifar100_content += '_verticaled'
                config = {
                    'rotation_range': rotation,
                    'zoom_range': zoom,
                    'horizontal_flip': horizontal,
                    'vertical_flip': vertical,
                }
                print('color:' + str(np.shape(x)))
                (x, y) = process.augment(x, y, config)
                print('augment:' + str(np.shape(x)))
                if channel == 1:
                    x = process.grayize(x)
                    print('gray:' + str(np.shape(x)))
            elif channel == 1:
                x = process.grayize(x)
            x = np.reshape(x, (nums.CIFAR100_TRAIN, width * height * channel))
            trainset[train_index: train_index + nums.CIFAR100_TRAIN, 0] = y
            trainset[train_index: train_index + nums.CIFAR100_TRAIN, 1:] = x
            train_index += nums.CIFAR100_TRAIN
            cifar100_content += '\n'
            print("CIFAR100训练集处理完成！")

            # 载入当前数据库数据
            print('正在处理CIFAR100测试集...')
            test_data = np.load(paths.PRODUCT_PATH + dataset + "_test.npy")
            test_data[:, 0] += 100
            x = test_data[:, 1:]
            x = np.reshape(x, (nums.CIFAR10_TEST, 32, 32, 3))
            if width != 32 or height != 32:
                x = process.resize(x, (width, height))
            # # 获取数据增强参数
            # if isaugment:
            #     if post.get("rotation_" + dataset) is not None:
            #         rotation = int(post.get("rotation_" + dataset))
            #         cifar10_content += '_rotate%d' % rotation
            #     else:
            #         rotation = 0
            #     if post.get("zoom_" + dataset) is not None:
            #         zoom = float(post.get("zoom_" + dataset))
            #         cifar10_content += '_zoom%f' % zoom
            #     else:
            #         zoom = 0
            #     horizontal = False
            #     vertical = False
            #     if post.get("horizontal_" + dataset) is not None:
            #         horizontal = True
            #         cifar10_content += '_horizontaled'
            #     if post.get("vertical_" + dataset) is not None:
            #         vertical = True
            #         cifar10_content += '_verticaled'
            #     config = {
            #         'rotation_range': rotation,
            #         'zoom_range': zoom,
            #         'horizontal_flip': horizontal,
            #         'vertical_flip': vertical,
            #     }
            #     print('color:' + str(np.shape(x)))
            #     (x, y) = process.augment(x, y, config)
            #     print('augment:' + str(np.shape(x)))
            #     if channel == 1:
            #         x = process.grayize(x)
            #         print('gray:' + str(np.shape(x)))
            # elif channel == 1:
            #     x = process.grayize(x)
            if channel == 1:
                x = process.grayize(x)
            x = np.reshape(x, (nums.CIFAR100_TEST, width * height * channel))
            testset[test_index: test_index + nums.CIFAR100_TEST, 0] = test_data[:, 0]
            testset[test_index: test_index + nums.CIFAR100_TEST, 1:] = x
            test_index += nums.CIFAR100_TEST
            print("CIFAR100测试集处理完成！")
    np.save(train_outf, trainset)
    np.save(test_outf, testset)
    content = header + mnist_content + svhn_content + cifar10_content + cifar100_content
    sql = "UPDATE task SET train_path='%s', test_path='%s', state=%d, content='%s' WHERE create_time='%s'" % (train_outf_name, test_outf_name, 1, content, str(creat_time))
    cursor = connection.cursor()
    cursor.execute(sql)
    print("保存成功！")








