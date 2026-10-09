"""Informative coarse-to-fine patch selection for both LoD branches.

The four-child indexing formula is preserved from utils.get_index; it is not
replaced with a different patch-selection algorithm. See IMPLEMENTATION.md.
"""

import torch


def gather_tokens(tokens, indices):
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def source_fine_indices(indices, grid):
    first = 4 * indices - 2 * indices.remainder(grid)
    return torch.cat((first, first + 1, first + grid, first + grid + 1), dim=1)


def spatial_fine_indices(indices, coarse_grid, fine_grid):
    """Map each coarse cell to every child cell of the actual fine grid.

    The source's four-child formula is retained separately. This generalized
    mapping is opt-in and is not claimed to reproduce the reported experiment.
    """
    if coarse_grid < 1 or fine_grid % coarse_grid:
        raise ValueError("Fine and coarse grids must have an integer ratio.")
    scale = fine_grid // coarse_grid
    offsets = torch.arange(scale, device=indices.device)
    rows = indices.div(coarse_grid, rounding_mode="floor") * scale
    columns = indices.remainder(coarse_grid) * scale
    children = (rows[..., None, None] + offsets[None, None, :, None]) * fine_grid
    children = children + columns[..., None, None] + offsets[None, None, None, :]
    return children.flatten(1)
