import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import get_model


def test_model_output_shape():
    """Model should map a batch of CIFAR images to (batch, num_classes) logits."""
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    dummy = torch.randn(4, 3, 32, 32)  # batch of 4 CIFAR-sized RGB images
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (4, 10)


def test_unsupported_architecture_raises():
    try:
        get_model(architecture="vgg16", num_classes=10)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unsupported architecture")
