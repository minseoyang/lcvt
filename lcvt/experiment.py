"""Shared configuration, checkpoint and four-head evaluation routines."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import LCvT, LCvTConfig


HEADS = ("lod1_coarse", "lod1_fine", "lod2_coarse", "lod2_fine")


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    model = LCvTConfig(**values["model"])
    training = values.get("training", {})
    return model, training


def flat_heads(outputs):
    return [outputs[level][granularity] for level in ("lod1", "lod2") for granularity in ("coarse", "fine")]


def classification_loss(outputs, broad, fine, weights=(1.0, 1.0, 1.0, 1.0)):
    if len(weights) != 4 or any(w < 0 for w in weights) or not any(weights):
        raise ValueError("Provide four nonnegative loss weights with a positive total.")
    targets = (broad, broad, fine, fine)
    return sum(w * F.cross_entropy(logits, label) for w, logits, label in zip(weights, flat_heads(outputs), targets))


@torch.no_grad()
def evaluate(model, loader, device, weights=(1.0, 1.0, 1.0, 1.0)):
    model.eval()
    count, loss_sum = 0, 0.0
    correct = [0] * 4
    for images, broad, fine in loader:
        images, broad, fine = [t.to(device) for t in (images, broad, fine)]
        outputs = model(images)
        batch = images.shape[0]
        loss_sum += classification_loss(outputs, broad, fine, weights).item() * batch
        for i, (logits, labels) in enumerate(zip(flat_heads(outputs), (broad, broad, fine, fine))):
            correct[i] += (logits.argmax(-1) == labels).sum().item()
        count += batch
    if not count:
        raise ValueError("The evaluation loader is empty.")
    return {"samples": count, "loss": loss_sum / count, "top1_percent": {
        name: 100 * value / count for name, value in zip(HEADS, correct)
    }, "early_exit": False}


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "lcvt-experiment-v1":
        raise ValueError("Expected a curated LCvT checkpoint. Original research checkpoints need migration and validation.")
    model = LCvT(LCvTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
