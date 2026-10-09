"""Portable image manifests and an explicit fine-to-broad label tree."""

import csv
from pathlib import Path, PurePosixPath

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def read_manifest(path, num_classes):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "lod1", "lod2"]:
            raise ValueError("CSV header must be path,lod1,lod2.")
        rows, seen, parents = [], set(), {}
        for line, row in enumerate(reader, 2):
            path = PurePosixPath(row["path"].replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in str(path) or str(path) == ".":
                raise ValueError(f"Line {line}: image paths must be relative to the data root.")
            broad, fine = int(row["lod1"]), int(row["lod2"])
            if not 0 <= broad < num_classes[0] or not 0 <= fine < num_classes[1]:
                raise ValueError(f"Line {line}: labels are outside the configured class ranges.")
            if str(path) in seen:
                raise ValueError(f"Line {line}: duplicate image path.")
            if fine in parents and parents[fine] != broad:
                raise ValueError(f"Line {line}: a fine class has conflicting broad parents.")
            rows.append((str(path), broad, fine))
            seen.add(str(path))
            parents[fine] = broad
    if not rows:
        raise ValueError("The manifest is empty.")
    return rows


def image_transform(image_size, training=False):
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        # Retained from the supplied CompCars training script.
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(30),
                  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]
    steps += [transforms.ToTensor(), transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )]
    return transforms.Compose(steps)


class HierarchicalImages(Dataset):
    def __init__(self, manifest, root, config, training=False):
        self.rows = read_manifest(manifest, config.num_classes)
        self.root = Path(root).resolve()
        self.transform = image_transform(config.image_size, training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        relative, broad, fine = self.rows[index]
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path resolves outside the data root.")
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, broad, fine


def validate_splits(train_rows, validation_rows):
    if {r[0] for r in train_rows} & {r[0] for r in validation_rows}:
        raise ValueError("Training and validation manifests overlap.")
    parents = {}
    for _, broad, fine in train_rows + validation_rows:
        if fine in parents and parents[fine] != broad:
            raise ValueError("A fine class has conflicting parents across splits.")
        parents[fine] = broad


def parent_labels(fine_predictions, mapping, num_classes):
    """Derive LoD 1 labels from LoD 2 predictions without another branch run."""
    if len(mapping) != num_classes[1] or any(not isinstance(v, int) or not 0 <= v < num_classes[0] for v in mapping):
        raise ValueError("The hierarchy must contain one valid LoD 1 parent per LoD 2 class.")
    parents = torch.tensor(mapping, device=fine_predictions.device, dtype=torch.long)
    return parents[fine_predictions]
