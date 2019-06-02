import  resources.paths as paths


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

dir_path = paths.MNIST_PATH
train_path = dir_path + 'train-images-idx3-ubyte'
train_label = dir_path + 'train-labels-idx1-ubyte'
test_path = dir_path + 't10k-images-idx3-ubyte'
test_label = dir_path + 't10k-labels-idx1-ubyte'

path = paths.MNIST_PATH
convert(train_path, train_label,
        path + "mnist_train.csv", 60000)
convert(test_path, test_label,
        path + "mnist_test.csv", 10000)