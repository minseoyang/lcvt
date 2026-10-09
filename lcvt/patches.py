"""Helpers for the source's optional LoD 1 selection path (disabled by default).

The four-child indexing formula is preserved from utils.get_index; it is not
replaced with a different patch-selection algorithm. See IMPLEMENTATION.md.
"""

import torch


def gather_tokens(tokens, indices):
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def source_fine_indices(indices, grid):
    first = 4 * indices - 2 * indices.remainder(grid)
    return torch.cat((first, first + 1, first + grid, first + grid + 1), dim=1)
