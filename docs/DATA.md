# Data and label hierarchy

## CompCars

Download images from the [official CompCars dataset](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) according to its access and usage conditions. This repository distributes **path/label manifests only**, not images.

CSV format:

```csv
path,lod1,lod2
78/1/2014/7c68748003aa0c.jpg,32,0
```

`path` is relative to the dataset image root. `lod1` and `lod2` are zero-based integer IDs: 75 manufacturers and 431 car models. Labels are preserved from the supplied lists; **do not derive them directly from the numeric folder names**. For example, manufacturer directory `78` maps to class ID `32` in the supplied labeling.

| Public manifest | Original list | Rows | Use in this refactor |
| --- | --- | ---: | --- |
| `train.csv` | `train.txt` | 36,334 | gradient updates |
| `validation.csv` | `valid.txt` | 5,307 | checkpoint selection |
| `evaluation.csv` | `test.txt` | 10,442 | held-out final evaluation |

Audit: no duplicate image paths within a split, no shared paths between the three splits, and no conflicting manufacturer parents for the 431 model IDs. `metadata.json` contains manifest hashes and counts. These checks concern label files; image availability/content and exact original experimental use remain unverified.

**Split naming discrepancy:** Section 5.1 calls the 10,442-image split “validation,” matching the size of the supplied `test.txt`. The supplied training script also uses a different 5,307-image `valid.txt`. Both are preserved separately here. The refactor selects checkpoints on the 5,307-image split and keeps the 10,442-image split for final evaluation. This is an explicit protocol choice, not proof of the original paper protocol.

`hierarchy.json` is a 431-element JSON array. Element `i` is the LoD 1 parent ID of LoD 2 class `i`, derived consistently from all three manifests. Human-readable manufacturer/model names are not reconstructed from numeric directory names.

## ImageNet

Obtain data from [ImageNet](https://www.image-net.org/). Provide your own verified `path,lod1,lod2` manifests and fine-to-broad hierarchy. The paper reports 664/1000 classes and 560,000/140,000 train/test images. Exact experiment manifests and the 664-class hierarchy were not identified in the supplied folder.

The local `prefix_to_lod1_mapping.json` has **61** entries grouped by the first four characters of synset IDs. It does not supply the paper's 664-class hierarchy and is intentionally excluded. A synset ID prefix is not sufficient to reconstruct semantic parent labels.

```bash
python train.py --config /path/to/verified_imagenet_config.json --data-root /path/to/imagenet --train-csv /path/to/train.csv --val-csv /path/to/validation.csv --output runs/imagenet
```

The loader is dataset-independent. Its default augmentation is taken from the supplied CompCars script. No inferred ImageNet preset is shipped: the original ImageNet branch has incomplete class configuration and references unavailable metadata. ImageNet model settings, augmentation and splitting must be confirmed before a reproduction run.
