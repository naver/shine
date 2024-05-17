import torch
import pathlib
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder


def load_imagenet_val(dataset_root, batch_size=32, num_workers=4, shuffle=True):
    dsclass = ImageNet
    data_dir = pathlib.Path(dataset_root)
    # Define the transformations
    tfms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset using ImageFolder
    dataset = dsclass(data_dir, split='val', transform=tfms)

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataset, dataloader







