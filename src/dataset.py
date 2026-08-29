import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_MEAN = [0.4914, 0.4822, 0.4465]
_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 2)

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader

