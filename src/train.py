import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataloaders  # noqa: E402
from model import get_model  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_config_path() -> str:
    env_path = os.environ.get("TRAINING_CONFIG")
    if env_path:
        return env_path
    for candidate in ("/app/configs/training_config.yaml", "configs/training_config.yaml"):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No training_config.yaml found (set TRAINING_CONFIG)")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def _log(entry: dict) -> None:
    print(json.dumps(entry), flush=True)


def main():
    config = load_config(resolve_config_path())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log({"event": "startup", "device": str(device)})

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
    )

    subset_n = int(os.environ.get("SUBSET", config["training"].get("subset", 0)))
    if subset_n > 0:
        train_loader = torch.utils.data.DataLoader(
            Subset(train_loader.dataset, range(min(subset_n, len(train_loader.dataset)))),
            batch_size=config["training"]["batch_size"], shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            Subset(val_loader.dataset, range(min(subset_n, len(val_loader.dataset)))),
            batch_size=config["training"]["batch_size"], shuffle=False,
        )
        _log({"event": "subset_mode", "samples": subset_n})

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        _log({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
            }, save_path)
            _log({"event": "checkpoint_saved", "path": str(save_path)})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _log({"event": "early_stopping", "epoch": epoch + 1})
                break

    _log({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)})


if __name__ == "__main__":
    main()
