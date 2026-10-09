# Verification and reproducibility

This repository includes the full two-LoD branch framework, including the restored LoD 2 selection path. Early exit is omitted. It has not been retrained on CompCars or ImageNet; the paper's accuracy and latency results have not been reproduced with this cleaned repository.

## Verified behavior

- **Eleven CPU tests** cover batch-one training and gradients in both fine-position embeddings, both selection paths, one fine pass per branch, source patch indexing, all combinations of the two selection flags, requested-LoD execution, progressive caching, explicit coarse/fine requests, checkpoint round-trip, data validation and geometry checks.
- High-confidence coarse predictions still execute both fine branches. There is no confidence-based exit option.
- Stage/branch hooks show that a LoD 1 request omits stages 2–3 and LoD 2, while a LoD 2 request omits the LoD 1 branch. Sequential cached requests run each required backbone stage only once.
- The optional spatial mapping covers the corresponding child cells and all fine patches, with finite inference at alpha 0, 0.5 and 1. It is not an accuracy reproduction.
- **Whole-model comparison** uses the complete `CvT_branch.py` branch implementation, with main-source heads 3/3/3 and depths 3/4/5 explicitly supplied to both models. Shared weights, eval mode, zero dropout, batch two and 256×256 RGB random inputs are used. All four selection combinations are checked against all four logits.
- Strict original-state conversion includes both LoD fine-position parameters. Only unused convolutional patch embeddings and duplicate FFN aliases are discarded.
- Synthetic-data training for one epoch, checkpoint saving, separate evaluation and image inference through the public CLIs.
- CompCars split counts, no duplicate/overlapping paths and a consistent complete 431-to-75 hierarchy.

Verified environment: Windows, Python 3.13.2, torch 2.6.0+cu118 and torchvision 0.21.0+cu118. Public tests and synthetic CLI checks run on CPU. The original model comparison runs on CUDA because its forward includes CUDA-only timers.

```bash
python -m unittest discover -s tests -v
```

The earlier verification against the uncommented `new_LCvT.py` loop concerned the initial incomplete release. It does not verify this restored framework. The current comparison instead checks the complete pre-exit branch sequence. Numerical agreement on random inputs does not establish real-dataset accuracy, stochastic training equivalence, original checkpoint provenance or latency equivalence.

## Requirements for a paper-result rerun

1. Confirm final architecture, optimizer and selection settings, given the paper/source differences and original false selection defaults.
2. Confirm CompCars split usage and obtain exact ImageNet splits and the 664-class hierarchy.
3. Identify the original trained checkpoint and complete training/evaluation record.
4. Evaluate all four outputs on real data using the chosen complete configuration.
5. Record hardware, batch size, warm-up, CUDA synchronization and cache lifetime for latency measurements.

The original timing lists are not adopted as benchmarks. The explicit cache implements the paper's LoD-transition workflow but does not establish the reported transition times.

Original weights are not uploaded. New/converted checkpoints use `lcvt-framework-v2` and strict tensor loading. The initial `lcvt-experiment-v1` schema is rejected because it omitted the LoD 2 fine-position parameter and used another fine path. Convert the original complete `state_dict` rather than silently changing an old curated checkpoint.
