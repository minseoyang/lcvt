"""Train all LoD heads; validation selection is separate from final evaluation."""

import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from lcvt import LCvT
from lcvt.data import HierarchicalImages, validate_splits
from lcvt.experiment import classification_loss, evaluate, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/compcars_source_dims.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-csv", default="data/compcars/train.csv")
    parser.add_argument("--val-csv", default="data/compcars/validation.csv")
    parser.add_argument("--output", default="runs/compcars")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="Override the configured number of epochs.")
    args = parser.parse_args()
    config, settings = load_config(args.config)
    seed = settings.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    train_data = HierarchicalImages(args.train_csv, args.data_root, config, training=True)
    val_data = HierarchicalImages(args.val_csv, args.data_root, config)
    validate_splits(train_data.rows, val_data.rows)
    batch_size = settings.get("batch_size", 32)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=args.workers)
    model = LCvT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.get("learning_rate", 1e-3),
                                 betas=tuple(settings.get("betas", (0.9, 0.999))),
                                 weight_decay=settings.get("weight_decay", 0.0))
    weights = settings.get("loss_weights", [1.0] * 4)
    epochs = args.epochs if args.epochs is not None else settings.get("epochs", 200)
    if epochs < 1 or batch_size < 1:
        parser.error("epochs and batch_size must be positive.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"model": config.to_dict(), "training": dict(settings, epochs=epochs)})
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        count, loss_sum = 0, 0.0
        for images, broad, fine in train_loader:
            images, broad, fine = [t.to(device) for t in (images, broad, fine)]
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(images), broad, fine, weights)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            count += images.shape[0]
        metrics = evaluate(model, val_loader, device, weights)
        row = {"epoch": epoch, "train_loss": loss_sum / count, "validation": metrics}
        write_json(output / f"epoch_{epoch:03d}.json", row)
        print(row, flush=True)
        checkpoint = {"format": "lcvt-curated-v1", "epoch": epoch,
                      "model_config": config.to_dict(), "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(), "training_config": settings,
                      "validation_metrics": metrics, "torch_version": str(torch.__version__)}
        torch.save(checkpoint, output / "last.pt")
        score = sum(metrics["top1_percent"][k] for k in ("lod1_fine", "lod2_fine")) / 2
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output / "best.pt")


if __name__ == "__main__":
    main()
