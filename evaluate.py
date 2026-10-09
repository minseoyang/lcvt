"""Report separate coarse/fine accuracies with early exit disabled."""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from lcvt.data import HierarchicalImages
from lcvt.experiment import evaluate, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--csv", default="data/compcars/evaluation.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="runs/evaluation.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    dataset = HierarchicalImages(args.csv, args.data_root, model.config)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    metrics = evaluate(model, loader, device, checkpoint["training_config"].get("loss_weights", [1.0] * 4))
    metrics["model_config"] = model.config.to_dict()
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
