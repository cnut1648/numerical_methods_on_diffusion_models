"""
Exec after you run infer.sh and saved inference results in inference_results/
Will produce FID score for each of the config in each of the dataset
"""
from collections import defaultdict
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import List
import os, json
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
import torch.utils.data as data

num_images = 101

mnist_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])
base_dir = 'inference_results'

def calc_FID(step_path: str, dataset: str, num_images: int) -> float:
    if dataset == "MNIST":
        test_dataset = MNIST(
            os.path.join(os.getcwd(), "temp", "mnist"),
            train=False,
            download=True,
            transform=mnist_transform,
        )
    elif dataset == "FMNIST":
        test_dataset = FashionMNIST(
            os.path.join(os.getcwd(), "temp", "f-mnist"),
            train=False,
            download=True,
            transform=mnist_transform,
        )
    else: # KMNIST
        test_dataset = KMNIST(
            os.path.join(os.getcwd(), "temp", "kmnist"),
            train=False,
            download=True,
            transform=mnist_transform,
        )
    fid = FrechetInceptionDistance().cuda()
    # for real
    dataloader = data.DataLoader(
        test_dataset, batch_size=num_images, shuffle=False
    )
    images, _ = next(iter(dataloader)) # only run once, since inference is also only @num_images
    images = images.repeat(1, 3, 1, 1) # grayscale to RGB
    images = images * 255 # scale to [0, 255]
    fid.update(images.type(torch.uint8).cuda(), real=True)
    
    # for generated
    images = []
    for img_file in sorted(os.listdir(step_path)):
        assert img_file.endswith('.png')
        path = os.path.join(step_path, img_file)
        image = Image.open(path).convert('L')
        image = mnist_transform(image)
        image = image.repeat(3, 1, 1) # grayscale to RGB
        images.append(image * 255) # scale to [0, 255]
    images = torch.stack(images).type(torch.uint8)
    fid.update(images.cuda(), real=False)
    return fid.compute().item()

for dataset in ['MNIST', 'KMNIST', 'FMNIST']:
    dataset_path = os.path.join(base_dir, dataset)
    results = defaultdict(dict)
    for config in os.listdir(dataset_path): # e.g. DDIM-F-PNDM
        config_path = os.path.join(dataset_path, config)
        if not os.path.isdir(config_path):
            continue
        model, method = config.split('-', 1)
        results[model][method] = {}
        for diffusion_step in ['600', '800', '1000']:
            step_path = os.path.join(config_path, diffusion_step)
            fid: float = calc_FID(step_path, dataset, num_images)
            results[model][method][diffusion_step] = fid

    with open(os.path.join(dataset_path, 'fid_results.json'), 'w') as f:
        json.dump(results, f, indent=4)