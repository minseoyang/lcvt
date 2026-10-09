"""LCvT with both LoD branches, selective refinement and LoD-aware inference.

The complete branch path is restored from new_LCvT.py and CvT_branch.py.
Early exit is omitted: fine inference always follows coarse inference.
"""

from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvStage, LoDBlock
from .patches import source_fine_indices, spatial_fine_indices, gather_tokens


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
    lod1_selection: bool = True
    lod2_selection: bool = True
    patch_mapping: str = "source"
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
        if self.patch_mapping not in ("source", "spatial"):
            raise ValueError("patch_mapping must be 'source' or 'spatial'.")
        if self.patch_mapping == "source" and self.coarse_scale < 2:
            raise ValueError("The original four-child mapping requires coarse_scale >= 2.")
        for side, patch in zip((self.image_size // 4, self.image_size // 16), self.patch_sizes):
            if patch < 1 or side % (patch * self.coarse_scale):
                raise ValueError("Each feature map must be divisible by patch_size * coarse_scale.")

    def to_dict(self):
        return asdict(self)


class LoDBranch(nn.Module):
    def __init__(self, channels, side, patch, classes, config, selection=True):
        super().__init__()
        self.patch = patch
        self.side = side
        self.coarse_side = side // config.coarse_scale
        self.coarse_grid = self.coarse_side // patch
        self.fine_grid = side // patch
        self.selection = selection
        self.patch_mapping = config.patch_mapping
        self.alpha = config.alpha
        self.ema_decay = config.ema_decay
        dim = config.branch_dim
        self.embedding = nn.Linear(channels * patch * patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.coarse_position = nn.Parameter(torch.randn(1, self.coarse_grid ** 2 + 1, dim))
        # Both branches need fine positions for informative-patch selection.
        self.fine_position = nn.Parameter(torch.randn(1, self.fine_grid ** 2 + 1, dim))
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
            important = ranking[:, :count]
            children = (
                source_fine_indices(important, self.fine_grid)
                if self.patch_mapping == "source"
                else spatial_fine_indices(important, self.coarse_grid, self.fine_grid)
            )
            cls = children.new_zeros(batch, 1)
            selected = gather_tokens(fine, torch.cat((cls, children + 1), 1))
            retained = gather_tokens(initial, ranking[:, count:] + 1)
            fine = torch.cat((selected, retained), 1)
        # CvT_branch.py has one fine pass per branch. The extra LoD 2 pass in
        # new_LCvT.py belonged to the added early-exit block and is removed.
        fine = self.embedding_dropout(fine)
        for block in self.fine_blocks:
            fine, _ = block(fine)
        return self.head(fine[:, 0])

    def forward(self, features):
        coarse, initial, encoded, scores = self.coarse(features)
        return {"coarse": coarse, "fine": self.fine(features, initial, encoded, scores)}


@dataclass
class InferenceCache:
    """Features for one fixed image batch and model in evaluation mode.

    Make a new cache for a new frame, changed weights or a changed device.
    This is an explicit in-memory API, not a DT server or tracking service.
    """

    owner: object = field(repr=False)
    image: torch.Tensor = field(repr=False)
    features: dict = field(default_factory=dict, repr=False)
    coarse_states: dict = field(default_factory=dict, repr=False)
    logits: dict = field(default_factory=dict, repr=False)


class LCvT(nn.Module):
    BRANCH_STAGES = {1: 0, 2: 2}

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
            LoDBranch(channels, side, patch, classes, c, selection)
            for channels, side, patch, classes, selection in zip(
                (c.channels[0], c.channels[2]), (c.image_size // 4, c.image_size // 16),
                c.patch_sizes, c.num_classes, (c.lod1_selection, c.lod2_selection),
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

    def forward(self, image, lods=(1, 2)):
        """Train all four heads by default, or execute only requested branches."""
        self._validate_image(image)
        requested = (lods,) if isinstance(lods, int) else tuple(lods)
        if not requested or len(set(requested)) != len(requested) or any(lod not in (1, 2) for lod in requested):
            raise ValueError("Request LoD 1, LoD 2, or both without duplicates.")
        last_stage = max(self.BRANCH_STAGES[lod] for lod in requested)
        outputs = {}
        features = image
        for index in range(last_stage + 1):
            features = self.stages[index](features)
            for lod in requested:
                if self.BRANCH_STAGES[lod] == index:
                    outputs[f"lod{lod}"] = self.branches[lod - 1](features)
        return outputs

    def _validate_prediction(self, lod, granularity):
        if self.training:
            raise RuntimeError("Call model.eval() before prediction.")
        if lod not in (1, 2) or granularity not in ("coarse", "fine"):
            raise ValueError("Use lod=1/2 and granularity='coarse'/'fine'.")

    @torch.no_grad()
    def prepare_cache(self, image):
        """Start a lazy cache for one image batch; no stages run yet."""
        self._validate_prediction(1, "fine")
        self._validate_image(image)
        return InferenceCache(self, image.detach().clone())

    @torch.no_grad()
    def predict_from_cache(self, cache, lod=1, granularity="fine"):
        """Continue a LoD request using already computed backbone features."""
        self._validate_prediction(lod, granularity)
        if not isinstance(cache, InferenceCache) or cache.owner is not self:
            raise ValueError("The cache must be created by this model.")
        key = (lod, granularity)
        if key in cache.logits:
            return cache.logits[key]
        target = self.BRANCH_STAGES[lod]
        for index in range(target + 1):
            if index not in cache.features:
                previous = cache.image if index == 0 else cache.features[index - 1]
                cache.features[index] = self.stages[index](previous)
        branch = self.branches[lod - 1]
        if lod not in cache.coarse_states:
            coarse, initial, encoded, scores = branch.coarse(cache.features[target])
            cache.coarse_states[lod] = (initial, encoded, scores)
            cache.logits[(lod, "coarse")] = coarse
        if granularity == "fine":
            cache.logits[key] = branch.fine(cache.features[target], *cache.coarse_states[lod])
        return cache.logits[key]

    @torch.no_grad()
    def predict(self, image, lod=1, granularity="fine"):
        """Execute the requested LoD branch, without confidence-based exits."""
        self._validate_prediction(lod, granularity)
        return self.predict_from_cache(self.prepare_cache(image), lod, granularity)
