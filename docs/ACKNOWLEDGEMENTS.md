# Acknowledgements and provenance

The research is by Min-Seo Yang, Ji-Wan Kim and Hyun-Suk Lee. Yang and Kim are marked as equal contributors in the [published article](https://doi.org/10.3390/electronics14193942). The implementation is curated from author-provided LCvT research code.

The architecture builds on:

- Wu et al., **CvT: Introducing Convolutions to Vision Transformers**, ICCV 2021 — convolutional token embedding and attention projection (paper reference 14).
- Chen et al., **CF-ViT: A General Coarse-to-Fine Method for Vision Transformer**, AAAI 2023 — coarse-to-fine refinement, class-attention patch selection and feature reuse (paper reference 19).
- Teerapittayanon et al., **BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks**, ICPR 2016 — early-exit mechanisms (paper reference 16).

Related references are listed in the [LCvT paper](https://www.mdpi.com/2079-9292/14/19/3942). No third-party model repository, distributed training utility file, pretrained weight or dataset image is vendored in this refactor. Layer and indexing operations are implemented directly with PyTorch.

The article is published under CC BY 4.0. That article license does not automatically determine a software license for this repository. No new software license is assigned here. Dataset access/use remains governed by the respective dataset providers.
