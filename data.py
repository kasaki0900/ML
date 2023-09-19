import torch
from torch import nn
from torch import tensor
from torch.nn import functional as f
from torchvision import datasets, transforms


def read_mnist():
    cache = {}
    mnist_data = datasets.MNIST(
        root='',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    print(mnist_data)

    cache['x_train'] = mnist_data.train_data
    cache['y_train'] = mnist_data.train_labels

    cache['x_test'] = mnist_data.test_data
    cache['y_test'] = mnist_data.test_labels

    return cache


def reshape(x):
    # resize one image with 2 dims to a vector.
    squeezed = x.clone().detach().float()
    data_shape = squeezed.shape

    data_size = data_shape[0]

    squeezed.resize_(data_size, data_shape[1]*data_shape[2])
    squeezed.request_grad = True

    return squeezed


def one_hot(y, classes=10):
    return f.one_hot(y, num_classes=classes).float()


def de_one_hot(y):
    # make the one-hot form results back to number labels
    return torch.argmax(y, dim=1)


def batch_partition(x, y, batch_size=5000):
    cache = {}
    batches_x = []
    batches_y = []

    set_size = x.shape[0]
    n = set_size // batch_size  # num of full batches
    leftover = set_size % n  # data left or not

    for i in range(n):
        batches_x.append(x[i:batch_size*(i+1), :])
        batches_y.append(y[i:batch_size*(i+1), :])
    if leftover:
        batches_x.append(x[batch_size*n:, :])
        batches_y.append(y[batch_size*n:, :])

    cache['batch_x'] = batches_x
    cache['batch_y'] = batches_y
    cache['batch_num'] = n + (1 if leftover else 0)

    return cache
