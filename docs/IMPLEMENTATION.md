# Implementation of the full LoD framework

The release includes both LoD branches, their coarse/fine paths, informative-patch selection and feature reuse. It restores the commented LoD 2 path in `new_LCvT.py` by cross-checking the complete implementation in `CvT_branch.py`. **Early exit is the only omitted inference mechanism.** The scope is the framework, not only the statements left uncommented in one source file.

## Framework components

| Component | Implementation | Provenance |
| --- | --- | --- |
| Shared convolutional transformer backbone | `ConvStage`, `ConvAttention` in `layers.py` | `new_LCvT.py` and `module.py` |
| LoD 1 branch after stage 1 | `LoDBranch`, `branches[0]` | both original model files |
| LoD 2 branch after stage 3 | `LoDBranch`, `branches[1]` | both original model files |
| Coarse class attention and EMA | `LoDBranch.coarse()` | both branches' coarse loops |
| Informative-patch ranking and mixed fine/coarse tokens | `LoDBranch.fine()`, `patches.py` | active LoD 1 and restored LoD 2 selection |
| Fine positional embeddings | `fine_position` in both branches | `LoD1_pos_emb_fine`, restored `LoD2_pos_emb_fine` |
| Coarse-to-fine feature transfer | `reuse` MLP and nearest interpolation | `reuse_block` / `reuse_block2` |
| Joint coarse/fine loss at both LoDs | `classification_loss()` | source training loop; paper Equation 9 |
| Requested-LoD execution | `forward(..., lods=...)`, `predict()` | source branch flags reorganized into a request API |
| Reuse across LoD transitions | `prepare_cache()`, `predict_from_cache()` | explicit API implementing the workflow in paper Section 4.1 |
| Lower-LoD label from a higher-LoD prediction | `parent_labels()`, `infer.py --hierarchy` | supplied class hierarchy; paper Section 2 |

See [the paper, Sections 2–4](https://www.mdpi.com/2079-9292/14/19/3942).

## Restored branch flow

Each branch performs coarse inference, accumulates class attention, ranks informative coarse patches, transfers coarse features to fine tokens, combines selected fine tokens with remaining coarse tokens, then performs fine inference. Fine inference always runs when requested; no confidence threshold decides whether to skip it.

Both `lod1_selection` and `lod2_selection` default to **true** in the framework release. The original constructor defaults were false. These flags are intentionally enabled to expose the restored mechanism; this is not a claim that the supplied training command enabled them. Each flag can be disabled independently for an ablation.

`new_LCvT.py` added an early-exit `if/else` inside LoD 2, but left another fine-encoder loop afterward. It also commented out LoD 2 selection and no longer constructed `Embedding_LoD2_fine`. Removing that decision block and restoring the complete sequence from `CvT_branch.py` yields **one coarse pass and one fine pass at each LoD**. The accidental duplicated fine pass is not retained.

## Patch-index conventions

The default `patch_mapping="source"` preserves the original `utils.get_index` four-child formula, including its child ordering. This enables direct comparison with the restored original branches. It selects four fine indices per informative coarse token. The original helper uses the fine-grid width in its calculation; it is not a general geometric mapping between arbitrary coarse and fine grids.

`patch_mapping="spatial"` is an explicit alternative that maps a selected coarse cell to all children of the actual fine grid. At the supplied 2×2 coarse / 8×8 fine geometry, a cell has 4×4 children. This follows the general splitting geometry described by the paper, but changes the original token sequence and requires its own training/evaluation. It is not silently substituted for the original helper or presented as an experimentally verified preset.

Both mappings retain unselected coarse tokens and include the class token. The spatial mapping is checked for correct regions, complete coverage and alpha values 0, 0.5 and 1.

## Preserved parameters and numerical operations

- Main-source backbone defaults: channels 64/64/64, depths 3/4/5, heads 3/3/3 and head dimensions 64.
- Branch dimension 384, six heads, three coarse/fine blocks, patch sizes 8 and 2, coarse scale 4, alpha 0.5 and EMA decay 0.99.
- Original nested residuals: `2*x + attention(x)`, then `2*x + MLP(x)`.
- Bilinear coarse-feature downsampling with `align_corners=True`, original patch flatten order and nearest-neighbor feature transfer.
- Original EMA target block indices 1–5; index 0 is skipped. Fine positions are used by the selection path, matching the complete original branch implementation.
- A classifier shared between coarse and fine inference within each LoD.

The complete `CvT_branch.py` has different backbone defaults (heads 1/3/3 and depths 3/3/3). Source comparisons explicitly use the main file's heads 3/3/3 and depths 3/4/5 in both models, rather than confusing constructor differences with a branch calculation difference.

## LoD-aware inference and caching

`predict(image, lod=1)` runs stage 1 and the LoD 1 branch. `predict(image, lod=2)` runs stages 1–3 and the LoD 2 branch, skipping the LoD 1 branch. `forward(image)` remains the joint-training path with four outputs; `forward(image, lods=(1,))` or `(2,)` limits the requested branches.

The cache is explicit, in memory and bound to one evaluation-mode model and one image snapshot. The caller creates a new cache for each new frame or changed model/device. A LoD 1→2 request reuses stage 1 and computes only the remaining stages and LoD 2 branch. A LoD 2→1 request can reuse the stored stage 1 feature; the hierarchy can also derive the lower-level label without another classifier call. Cached coarse results are reused when continuing to fine inference.

This cache API is refactor code implementing the paper's described workflow. It was not found as a complete tracking/server implementation in the original folder. No DT server, object tracking or persistent multi-object storage is included.

## Portability and checkpoint format

CUDA-only timing and debug prints are removed. Batch axes are retained, tensors follow the input device, unused imports are removed, and data paths are relative to a supplied root. Required layer operations are implemented directly with PyTorch.

Checkpoints use `lcvt-framework-v2`. Both fine-position parameters are required. The initial `lcvt-experiment-v1` release omitted the LoD 2 position tensor and used the duplicated LoD 2 loop, so it is rejected rather than loaded with missing parameters. Use `convert_checkpoint.py` on the original complete state dictionary. This conversion validates names and shapes; it does not prove experimental provenance or equivalent accuracy after changing selection settings.

## Configuration and reproduction limits

The published training parameters retain the supplied source defaults: SGD, learning rate 0.001, momentum 0.9, weight decay 0.0001, batch size 32, 301 epochs and equal loss weights. Seed 42, validation after every epoch and selection of `best.pt` by mean fine validation accuracy are explicit curation choices.

Paper Table 1 reports channels 64/192/384, depths 3/4/4 and heads 1/3/3; Section 5.1 reports Adam and 200 epochs. These differ from the main source defaults. The README reports the paper's results with attribution; no architecture/configuration is claimed to reproduce them without the final experiment record.

CNN/ResNet/ViT/CvT-only/LCvT-OC baselines, notebooks, dataset images, weights and private logs remain outside this focused LCvT repository.
