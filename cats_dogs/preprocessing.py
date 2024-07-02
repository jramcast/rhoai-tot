import os
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define transformations to be applied to each image
transform = transforms.Compose(
    [
        # Resize the images to a fixed size
        transforms.Resize((50, 50)),
        # Convert images to PyTorch tensors
        transforms.ToTensor(),
        # Normalize the images
        # Using the mean and std of ImageNet is a common practice
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def preprocess(data_dir: Union[str, os.PathLike]):
    """
    Returns a tuple of 3 elements.
    (train data loader, validation data loader, dataset)
    """
    # Delete corrupt JPG files
    delete_corrupted(data_dir)

    # Create the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, dataset


def delete_corrupted(dataset_dir: Union[str, os.PathLike]):
    num_skipped = 0
    for dir_name in ("Cat", "Dog"):
        dir_path = Path(dataset_dir) / dir_name
        for filepath in dir_path.glob("*.jpg"):
            with open(filepath, "rb") as f:
                is_jfif = b"JFIF" in f.peek(10)

            if not is_jfif:
                num_skipped += 1
                os.remove(filepath)

    print(f"Deleted {num_skipped} corrupted images.")
