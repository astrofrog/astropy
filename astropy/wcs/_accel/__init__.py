"""
Fast pixel-to-pixel transformations for zenithal WCS projections.

This module provides accelerated coordinate transformations between two WCS with
zenithal projections using an exact plane-to-plane algorithm: a per-projection
radial rescale to/from the gnomonic projection, sandwiching a single
gnomonic-to-gnomonic homography. The transform is exact (to floating-point
precision) for any pair of TAN, SIN, STG, ARC, ZEA projections sharing a
celestial frame, and reduces to a single 3x3 projective matrix for TAN -> TAN.
"""

from ._matrix import (
    PlaneToPlaneTransform,
    apply_transform,
    compute_transform,
    compute_transform_matrix,
)
from ._projections import SUPPORTED_PROJECTIONS

__all__ = [
    "compute_transform",
    "apply_transform",
    "PlaneToPlaneTransform",
    "compute_transform_matrix",
    "SUPPORTED_PROJECTIONS",
]
