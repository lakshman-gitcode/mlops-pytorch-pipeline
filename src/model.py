import torch.nn as nn
from torchvision import models


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = models.resnet18(weights=None, num_classes=num_classes)

    # CIFAR-10 stem adaptation
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model
