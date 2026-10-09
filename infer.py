"""Run the requested LoD branch for one image (no early exit)."""

import argparse
import json

from PIL import Image
import torch

from lcvt.data import image_transform, parent_labels
from lcvt.experiment import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--lod", type=int, choices=(1, 2), default=2)
    parser.add_argument("--granularity", choices=("coarse", "fine"), default="fine")
    parser.add_argument("--hierarchy", help="Optional JSON array: LoD 1 parent for each LoD 2 class.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, device)
    with Image.open(args.image) as image:
        inputs = image_transform(model.config.image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model.predict(inputs, args.lod, args.granularity)
    label = logits.argmax(-1)
    result = {"lod": args.lod, "granularity": args.granularity, "class_id": label.item(),
              "confidence": logits.softmax(-1).amax(-1).item()}
    if args.lod == 2 and args.hierarchy:
        with open(args.hierarchy, encoding="utf-8") as handle:
            mapping = json.load(handle)
        result["lod1_parent_id"] = parent_labels(label, mapping, model.config.num_classes).item()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
