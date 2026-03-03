"""
Fast pixel-to-pixel transformations for zenithal WCS projections.

This module provides accelerated coordinate transformations between two WCS
with zenithal projections using the Montage "plane-to-plane" algorithm.
"""

from ._core import (
    SUPPORTED_PROJECTIONS,
    apply_transform,
    apply_transform_grid,
    compute_transform_matrix,
)

__all__ = [
    "compute_transform_matrix",
    "apply_transform",
    "apply_transform_grid",
    "SUPPORTED_PROJECTIONS",
]
