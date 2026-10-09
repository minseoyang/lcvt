# Verification and reproducibility

This repository preserves the supplied active LCvT experiment computation with early exit excluded. It has not been retrained on CompCars or ImageNet; the paper's accuracy and latency results have not been reproduced with this cleaned repository.

## Verified

- Seven CPU behavior tests: batch-one forward/backward with gradients in all coarse/fine encoders, original default token counts and repeated LoD 2 fine execution, output selection, optional LoD 1 helper, checkpoint round-trip, labels/split/path validation and input geometry checks.
- **Whole-model comparison** against the supplied `new_LCvT.py`: identical weights, eval mode, dropout zero, early exit forced off, batch size two and 256×256 RGB input. Four outputs agree numerically for both the default disabled LoD 1 selection and its optional enabled path. Maximum absolute differences were below 5e-7 in the preparation run.
- Explicit checkpoint conversion maps all required source tensors and rejects incompatible/missing parameters. Unused conv embeddings, the unused LoD 2 fine-position parameter and duplicate FFN aliases are the only discarded source keys.
- One synthetic-data training epoch, checkpoint saving, separate evaluation and image inference through the public CLIs.
- CompCars split counts, no duplicate/overlapping paths, and a complete consistent 431-to-75 hierarchy.

Verified environment: Windows, Python 3.13.2, torch 2.6.0+cu118, torchvision 0.21.0+cu118. Public behavior tests and synthetic CLI checks ran on CPU. Original/cleaned full-model comparisons ran on CUDA because the original code uses CUDA-only timing APIs.

```bash
python -m unittest discover -s tests -v
```

Numerical agreement was tested using random inputs and weights. It does not establish original checkpoint provenance, real-image accuracy, stochastic training trajectory equivalence, mixed-precision equivalence or timing equivalence.

## Remaining requirements for a paper-result rerun

1. Confirm which architecture and optimizer settings were used, given the paper/source differences.
2. Confirm the CompCars split naming/protocol and obtain the exact ImageNet splits and 664-class hierarchy.
3. Identify the trained checkpoint and its complete training settings, seed, augmentation and evaluation protocol.
4. Evaluate the chosen checkpoint on the real dataset and record all four metrics separately.
5. Define hardware, batch size, warm-up, CUDA synchronization and measurement scope before comparing latency. The original timing lists omitted stage 1 and mixed measurement scopes; they are not reused as benchmarks.

The public code includes no early-exit option or LoD feature cache. `infer.py --lod` selects a returned head after all stages/heads execute. The paper's reported accuracy and timing comparisons also disabled early exit.

Original weights are not uploaded. `convert_checkpoint.py` accepts the original plain LCvT `state_dict` using `weights_only=True` and an explicit name mapping. Converted/new checkpoints use the `lcvt-experiment-v1` schema, and `load_model()` loads their tensors strictly.
