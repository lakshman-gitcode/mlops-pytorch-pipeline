import io
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import get_model  # noqa: E402
from dataset import _MEAN, _STD  # reuse exact training normalization  # noqa: E402

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

app = Flask(__name__)
_model = None
_device = torch.device("cpu")  # serving runs CPU-only in this deployment

_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def load_model():
    global _model
    ckpt_path = os.environ.get(
        "CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"
    )
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=_device)
    model = get_model(
        architecture=checkpoint.get("architecture", "resnet18"),
        num_classes=checkpoint.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model


@app.route("/health", methods=["GET"])
def health():
    if _model is None:
        return jsonify({"status": "model not loaded"}), 503
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "model not loaded"}), 503
    if "image" not in request.files:
        return jsonify({"error": "no image provided (field name 'image')"}), 400

    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    ranked = sorted(
        ({"class": c, "probability": round(p, 4)} for c, p in zip(CIFAR10_CLASSES, probs)),
        key=lambda x: x["probability"], reverse=True,
    )
    return jsonify({"predictions": ranked, "top_class": ranked[0]["class"]}), 200


# Load at import time so gunicorn/flask workers are ready before first request.
load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
