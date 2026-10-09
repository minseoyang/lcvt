# Implementation and publication scope

The source of this refactor is the supplied `LCvT/new_LCvT.py`, with the required operations from `module.py`, `utils.py` and the CompCars path in `run_lcvt.py`. The publication scope follows the author's clarification: **use uncommented experiment code and exclude early exit**, which was not used in the reported experiments.

## Included computation

- Three shared CvT stages: convolutional embedding, depthwise-separable Q/K/V projection, attention and MLP.
- LoD 1 branch after stage 1; LoD 2 branch after stage 3.
- Coarse and fine encoders with a shared classification head at each LoD.
- Linear patch embedding, bilinear coarse-map downsampling with `align_corners=True`, and nearest-neighbor reuse of encoded coarse patch features.
- Four-head cross-entropy training; weights are explicit and default to one, reproducing the source loss sum.
- The uncommented optional LoD 1 selection code and its helper; **disabled by default**, as in the supplied run.

Early-exit decisions and flags are absent from the public model/CLI. Commented LoD 2 selection and unused branch conv embeddings are not included. There is no DT server or object-tracking system here.

## Preserved source behavior

| Source operation | Public representation |
| --- | --- |
| Backbone defaults | channels 64/64/64, blocks 3/4/5, heads 3/3/3, head dimension 64 |
| Branch defaults | dimension 384, six heads, three coarse/fine blocks, patch sizes 8 and 2 |
| Coarse/fine grids for 256-pixel input | 2×2 and 8×8 grids at both LoDs |
| Nested MSA/FFN residual additions | `residual_multiplier=2.0`: `2*x + attention(x)`, then `2*x + MLP(x)` |
| LoD 1 fine pass | one pass through its three fine blocks |
| LoD 2 non-exit fine path | two passes through the **same** three fine blocks, with dropout before each pass |
| Fine position embeddings when selection is off | not added, as in the active default source path |
| Optional LoD 1 attention aggregation | target block indices 1–5; index 0 is skipped; EMA decay 0.99 |
| Optional LoD 1 child-index helper | original four-child formula, not replaced with a different algorithm |

The repeated LoD 2 pass and nested residuals may deserve a separate architecture revision, but silently changing them during code organization would alter checkpoint outputs. They are intentionally retained.

The optional selection helper does not implement a general 2×2-to-8×8 spatial split. It uses the source's existing indexing formula. Because the experiment default disables selection, the public config retains `lod1_selection=false`; the repository does not present this helper as a validated general patch-selection method.

## Changes limited to organization and execution

- Remove commented experiments, unused conv embeddings, unused imports and debug prints.
- Replace einops/timm dependencies with PyTorch reshapes and initialization; preserve patch flatten order `(patch row, patch column, channel)`.
- Retain the batch axis instead of calling `squeeze()` on a resized feature map.
- Replace hard-coded `.cuda()` zero tensors with tensors on the input's device.
- Remove CUDA-only timing from model forward and expose named logits instead of timing lists.
- Separate model, data, loss, training, evaluation and inference; remove import-time argument parsing and TensorBoard creation.
- Replace absolute machine paths with data-root arguments and relative CSV manifests.
- Correct the **names** of LoD 2 coarse/fine outputs in training metrics. The equal-weight source loss sum is unchanged by their original order swap.
- Provide an explicit, strict source-checkpoint key conversion rather than accepting partial weights silently.

The direct comparison used the original model with its early-exit condition forced off. With shared weights, eval mode, dropout disabled and two random RGB inputs at 256×256, all four outputs agreed within 1e-5 relative / 3e-6 absolute tolerance, both with the default LoD 1 selection flag and with the optional flag enabled. This verifies the refactor under those conditions; it is not a real-dataset performance reproduction.

## Training configuration provenance

`configs/compcars.json` retains source argument defaults: SGD, learning rate 0.001, momentum 0.9, weight decay 0.0001, batch size 32, 301 epochs and equal loss weights. The source declares schedule/gamma arguments but does not apply them in the inspected training loop, so this refactor does not invent a scheduler.

Seed 42, per-epoch validation/checkpoint logging and choosing `best.pt` by the mean of the two fine validation accuracies are explicit organization defaults. The original script evaluated less frequently. Real-dataset reruns should record these protocol choices.

## Paper/source discrepancies

Table 1 reports backbone channels **64/192/384**, depths **3/4/4**, and heads **1/3/3**, whereas the supplied source defaults are **64/64/64**, **3/4/5**, **3/3/3**. Section 5.1 reports Adam and 200 epochs, while the supplied training defaults use SGD and 301 epochs. These discrepancies are documented rather than resolved by constructing another model and labeling it the experiment code.

The paper's broader conceptual framework includes selective refinement, early exit and reuse across dynamic LoD requests. This minimal release covers the supplied active experiment computation; it does not restore commented/unverified paths to match every proposed mechanism.

## Excluded files

CNN/ResNet/ViT/CvT-only/LCvT-OC baselines, previous model copies, notebooks, Grad-CAM tools and outputs, dataset images, pretrained checkpoints, training logs, full environment exports and copied distributed training utilities are excluded.
