"""LCvT experiment core, preserving the supplied active non-exit computation.

Commented LoD 2 selection and early exit are excluded. See IMPLEMENTATION.md
for the intentionally retained residual and repeated-fine-pass conventions.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvStage, LoDBlock
from .patches import source_fine_indices, gather_tokens


@dataclass
class LCvTConfig:
    image_size: int = 256
    num_classes: tuple = (75, 431)
    # Defaults retain the supplied source dimensions, not paper Table 1.
    channels: tuple = (64, 64, 64)
    depths: tuple = (3, 4, 5)
    heads: tuple = (3, 3, 3)
    head_dims: tuple = (64, 64, 64)
    branch_dim: int = 384
    branch_heads: int = 6
    branch_depth: int = 3
    patch_sizes: tuple = (8, 2)
    coarse_scale: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    embedding_dropout: float = 0.1
    lod1_selection: bool = False
    alpha: float = 0.5
    ema_decay: float = 0.99
    residual_multiplier: float = 2.0

    def __post_init__(self):
        for name in ("channels", "depths", "heads", "head_dims"):
            values = getattr(self, name)
            if len(values) != 3 or any(v < 1 for v in values):
                raise ValueError(f"{name} must contain three positive integers.")
        if len(self.num_classes) != 2 or any(n < 1 for n in self.num_classes):
            raise ValueError("Two positive class counts are required.")
        if len(self.patch_sizes) != 2:
            raise ValueError("Two branch settings are required.")
        if self.image_size < 16 or self.image_size % 16 or self.coarse_scale < 1:
            raise ValueError("image_size must be a multiple of 16; coarse_scale must be positive.")
        if self.branch_dim < 1 or self.branch_heads < 1 or self.branch_dim % self.branch_heads:
            raise ValueError("branch_dim must be divisible by branch_heads.")
        if self.branch_depth < 1 or self.mlp_ratio < 1:
            raise ValueError("Encoder depths and MLP ratio must be positive.")
        if not 0 <= self.alpha <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("alpha must be in [0, 1]; ema_decay must be in [0, 1).")
        for side, patch in zip((self.image_size // 4, self.image_size // 16), self.patch_sizes):
            if patch < 1 or side % (patch * self.coarse_scale):
                raise ValueError("Each feature map must be divisible by patch_size * coarse_scale.")

    def to_dict(self):
        return asdict(self)


class LoDBranch(nn.Module):
    def __init__(self, channels, side, patch, classes, config, selection=False, fine_repeats=1):
        super().__init__()
        self.patch = patch
        self.side = side
        self.coarse_side = side // config.coarse_scale
        self.coarse_grid = self.coarse_side // patch
        self.fine_grid = side // patch
        self.selection = selection
        self.fine_repeats = fine_repeats
        self.alpha = config.alpha
        self.ema_decay = config.ema_decay
        dim = config.branch_dim
        self.embedding = nn.Linear(channels * patch * patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.coarse_position = nn.Parameter(torch.randn(1, self.coarse_grid ** 2 + 1, dim))
        # Only LoD 1's non-commented optional path uses fine positions.
        self.fine_position = nn.Parameter(torch.randn(1, self.fine_grid ** 2 + 1, dim)) if fine_repeats == 1 else None
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.coarse_blocks = nn.ModuleList([
            LoDBlock(dim, config.branch_heads, config.mlp_ratio, config.dropout, config.residual_multiplier)
            for _ in range(config.branch_depth)
        ])
        self.fine_blocks = nn.ModuleList([
            LoDBlock(dim, config.branch_heads, config.mlp_ratio, config.dropout, config.residual_multiplier)
            for _ in range(config.branch_depth)
        ])
        self.reuse = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * config.mlp_ratio), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim * config.mlp_ratio, dim), nn.Dropout(config.dropout),
        )
        # A classifier is shared between coarse and fine, as in the source.
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))

    def embed(self, features):
        batch, channels, height, width = features.shape
        p = self.patch
        # Keep the source flatten order: (patch row, patch column, channel).
        patches = features.reshape(batch, channels, height // p, p, width // p, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, p * p * channels)
        tokens = self.embedding(patches)
        return torch.cat((self.cls_token.expand(batch, -1, -1), tokens), 1)

    def coarse(self, features):
        reduced = F.interpolate(features, size=(self.coarse_side, self.coarse_side), mode="bilinear", align_corners=True)
        initial = self.embedding_dropout(self.embed(reduced) + self.coarse_position)
        tokens = initial
        scores = tokens.new_zeros(tokens.shape[0], self.coarse_grid ** 2)
        for index, block in enumerate(self.coarse_blocks):
            tokens, attention = block(tokens)
            # Preserve target_index=[1,2,3,4,5] from the supplied code.
            if index in (1, 2, 3, 4, 5):
                class_attention = attention.mean(1)[:, 0, 1:]
                scores = self.ema_decay * scores + (1 - self.ema_decay) * class_attention
        return self.head(tokens[:, 0]), initial, tokens, scores

    def fine(self, features, initial, encoded, scores):
        fine = self.embed(features)
        batch, _, dim = encoded.shape
        reuse = self.reuse(encoded[:, 1:]).transpose(1, 2).reshape(batch, dim, self.coarse_grid, self.coarse_grid)
        reuse = F.interpolate(reuse, size=(self.fine_grid, self.fine_grid), mode="nearest").flatten(2).transpose(1, 2)
        fine = fine + torch.cat((reuse.new_zeros(batch, 1, dim), reuse), 1)
        if self.selection:
            count = math.ceil(self.alpha * self.coarse_grid ** 2)
            ranking = scores.argsort(1, descending=True)
            fine = fine + self.fine_position
            children = source_fine_indices(ranking[:, :count], self.fine_grid)
            cls = children.new_zeros(batch, 1)
            selected = gather_tokens(fine, torch.cat((cls, children + 1), 1))
            retained = gather_tokens(initial, ranking[:, count:] + 1)
            fine = torch.cat((selected, retained), 1)
        # LoD 2's active non-exit path applies the same fine blocks twice.
        # Keep that behavior rather than silently changing checkpoint outputs.
        for _ in range(self.fine_repeats):
            fine = self.embedding_dropout(fine)
            for block in self.fine_blocks:
                fine, _ = block(fine)
        return self.head(fine[:, 0])

    def forward(self, features):
        coarse, initial, encoded, scores = self.coarse(features)
        return {"coarse": coarse, "fine": self.fine(features, initial, encoded, scores)}

class LCvT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LCvTConfig()
        c = self.config
        self.stages = nn.ModuleList()
        input_channels = 3
        for i, (kernel, stride, padding) in enumerate(((7, 4, 2), (3, 2, 1), (3, 2, 1))):
            self.stages.append(ConvStage(
                input_channels, c.channels[i], kernel, stride, padding, c.depths[i],
                c.heads[i], c.head_dims[i], c.mlp_ratio, c.dropout,
            ))
            input_channels = c.channels[i]
        self.branches = nn.ModuleList([
            LoDBranch(channels, side, patch, classes, c, selection, repeats)
            for channels, side, patch, classes, selection, repeats in zip(
                (c.channels[0], c.channels[2]), (c.image_size // 4, c.image_size // 16),
                c.patch_sizes, c.num_classes, (c.lod1_selection, False), (1, 2),
            )
        ])
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_image(self, image):
        expected = (3, self.config.image_size, self.config.image_size)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected or image.shape[0] < 1:
            raise ValueError(f"Expected non-empty [B, {expected[0]}, {expected[1]}, {expected[2]}] input.")

    def forward(self, image):
        """Compute the four experiment outputs, with no early-exit branch."""
        self._validate_image(image)
        stage1 = self.stages[0](image)
        lod1 = self.branches[0](stage1)
        stage3 = self.stages[2](self.stages[1](stage1))
        return {"lod1": lod1, "lod2": self.branches[1](stage3)}

    @torch.no_grad()
    def predict(self, image, lod=1, granularity="fine"):
        """Select an experiment output; all stages/heads still execute."""
        if self.training:
            raise RuntimeError("Call model.eval() before prediction.")
        if lod not in (1, 2) or granularity not in ("coarse", "fine"):
            raise ValueError("Use lod=1/2 and granularity='coarse'/'fine'.")
        return self(image)[f"lod{lod}"][granularity]
