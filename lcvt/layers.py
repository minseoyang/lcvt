"""Convolutional attention and source-derived LoD encoder operations.

Implemented with PyTorch tensor operations; no external CvT/DeiT utility files
are vendored. See docs/IMPLEMENTATION.md for the retained residual convention.
"""

import torch
from torch import nn


class ConvProjection(nn.Sequential):
    def __init__(self, dim, inner_dim, stride):
        super().__init__(
            nn.Conv2d(dim, dim, 3, stride, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, 1),
        )


class ConvAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        self.heads = heads
        self.scale = head_dim ** -0.5
        inner_dim = heads * head_dim
        self.query = ConvProjection(dim, inner_dim, 1)
        self.key = ConvProjection(dim, inner_dim, 2)
        self.value = ConvProjection(dim, inner_dim, 2)
        self.output = (
            nn.Identity() if heads == 1 and head_dim == dim
            else nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        )

    def _tokens(self, x):
        batch, channels, height, width = x.shape
        return x.reshape(batch, self.heads, channels // self.heads, height * width).transpose(2, 3)

    def forward(self, x, height, width):
        batch, _, dim = x.shape
        spatial = x.transpose(1, 2).reshape(batch, dim, height, width)
        q, k, v = [self._tokens(layer(spatial)) for layer in (self.query, self.key, self.value)]
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, height * width, -1)
        return self.output(output)


class ConvBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_ratio, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = ConvAttention(dim, heads, head_dim, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x, height, width):
        x = x + self.attention(self.attention_norm(x), height, width)
        return x + self.mlp(self.mlp_norm(x))


class ConvStage(nn.Module):
    def __init__(self, in_channels, dim, kernel, stride, padding, depth, heads, head_dim, mlp_ratio, dropout):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels, dim, kernel, stride, padding)
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            ConvBlock(dim, heads, head_dim, mlp_ratio, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.embedding(x)
        batch, channels, height, width = x.shape
        tokens = self.norm(x.flatten(2).transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens, height, width)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class LoDBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout, residual_multiplier=2.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.residual_multiplier = residual_multiplier
        self.attention_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        # The supplied MSA uses an identity output projection (drop_hidden=True).
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, count, dim = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(
            batch, count, 3, self.heads, dim // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        attention = self.attention_dropout(attention)
        output = (attention @ v).transpose(1, 2).reshape(batch, count, dim)
        # Preserve the source's nested residual additions explicitly.
        x = self.residual_multiplier * x + output
        x = self.residual_multiplier * x + self.mlp(x)
        return x, attention
