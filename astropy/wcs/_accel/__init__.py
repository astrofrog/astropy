"""
Fast pixel-to-pixel transformations for zenithal WCS projections.

This module provides accelerated coordinate transformations between two WCS
with zenithal projections using the Montage "plane-to-plane" algorithm.
"""

from ._matrix import apply_transform, compute_transform_matrix
from ._projections import SUPPORTED_PROJECTIONS

__all__ = [
    "compute_transform_matrix",
    "apply_transform",
    "SUPPORTED_PROJECTIONS",
]
