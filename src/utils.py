import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST


def get_dataset(dataset, image_size):
    mnist_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    if dataset == "MNIST":
        dataset = MNIST(
            os.path.join(os.getcwd(), "temp", "mnist"),
            train=True,
            download=True,
            transform=mnist_transform,
        )
        test_dataset = MNIST(
            os.path.join(os.getcwd(), "temp", "mnist"),
            train=False,
            download=True,
            transform=mnist_transform,
        )
    elif dataset == "FMNIST":
        dataset = FashionMNIST(
            os.path.join(os.getcwd(), "temp", "f-mnist"),
            train=True,
            download=True,
            transform=mnist_transform,
        )
        test_dataset = FashionMNIST(
            os.path.join(os.getcwd(), "temp", "f-mnist"),
            train=False,
            download=True,
            transform=mnist_transform,
        )
    else: # KMNIST
        dataset = KMNIST(
            os.path.join(os.getcwd(), "temp", "kmnist"),
            train=True,
            download=True,
            transform=mnist_transform,
        )
        test_dataset = KMNIST(
            os.path.join(os.getcwd(), "temp", "kmnist"),
            train=False,
            download=True,
            transform=mnist_transform,
        )

    return dataset, test_dataset
def inverse_data_transform(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)
