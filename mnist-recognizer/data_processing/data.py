import torch
from torch import nn
from torch import tensor
from torch.nn import functional as f
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class DataCache:
    # The train data_processing may need to be batch processed and shuffled, using loader can do faster.
    # Test data_processing do not.
    def __init__(self, train_data_loader: DataLoader, train_data: Dataset, test_data: Dataset):
        self.train_data_loader = train_data_loader
        self.test_data = test_data
        self.train_data = train_data


def read_mnist(batch_size=64):
    train_data = datasets.MNIST(
        root='',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    test_data = datasets.MNIST(
        root='',
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    # Don't put data into cuda before putting it into loader.

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    # Data in this loader have been float form and in cuda yet.

    return DataCache(train_data_loader, train_data, test_data)

