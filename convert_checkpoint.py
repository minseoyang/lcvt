"""Convert an original LCvT state_dict to the complete branch key layout."""

import argparse
from pathlib import Path

import torch

from lcvt import LCvT
from lcvt.checkpoint import import_source_state
from lcvt.experiment import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default="configs/compcars.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, training = load_config(args.config)
    model = LCvT(config)
    state = torch.load(args.source, map_location="cpu", weights_only=True)
    ignored = import_source_state(model, state)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format":"lcvt-framework-v2", "model_config":config.to_dict(),
                "model_state":model.state_dict(), "training_config":training,
                "converted_source_name":Path(args.source).name,
                "ignored_inactive_or_alias_keys":ignored}, destination)
    print(f"Converted {len(model.state_dict())} tensors; ignored {len(ignored)} inactive/alias keys.")
    print("Conversion does not establish the checkpoint's training provenance or paper-result equivalence.")


if __name__ == "__main__":
    main()
