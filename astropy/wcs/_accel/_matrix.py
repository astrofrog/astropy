"""
Plane-to-plane transform computation and application.

This module builds the exact plane-to-plane map between two zenithal WCS and
applies it to pixel coordinates. The map has three ingredients:

1. an affine pixel <-> intermediate map per WCS (the CD/PC matrix and CRPIX);
2. a per-projection radial rescale that converts a zenithal projection to/from
   the gnomonic (tangent-plane) projection, exactly (see ``_projections``);
3. the gnomonic-to-gnomonic homography ``Q = B2^T B1``, a pure rotation of the
   two tangent frames fixed entirely by the two reference points.

The expensive trigonometry of a full spherical round trip is replaced by a
single matrix multiply plus two scalar radial rescales per pixel; everything
that does not depend on the pixel is precomputed once in
``compute_transform``.

For the ``TAN`` -> ``TAN`` case both rescales are the identity, so the whole map
collapses to a single 3x3 projective matrix (``_compute_matrix_analytical``),
which is used as a fast path.
"""

import numpy as np

from ._projections import deproject_to_gnomonic, reproject_from_gnomonic
from ._wcs import (
    get_cd_matrix,
    get_tangent_basis,
    validate_same_frame,
    validate_wcs,
)


def _compute_matrix_analytical(wcs1, wcs2):
    """
    Compute the exact TAN -> TAN transformation matrix analytically.

    This is a fully analytical computation using only WCS header parameters
    (CRVAL, CRPIX, CD/PC+CDELT). No WCS coordinate conversions are performed.
    It is the special case of the general algorithm in which both radial
    rescales are the identity.
    """
    deg2rad = np.pi / 180.0

    cd1 = get_cd_matrix(wcs1)
    cd2 = get_cd_matrix(wcs2)

    # Convert CRPIX from FITS 1-indexed to 0-indexed pixel coordinates
    crpix1_0 = wcs1.wcs.crpix - 1
    crpix2_0 = wcs2.wcs.crpix - 1

    n1, ex1, ey1 = get_tangent_basis(wcs1)
    n2, ex2, ey2 = get_tangent_basis(wcs2)

    offset1 = -cd1 @ crpix1_0

    # P1: pixel1 -> intermediate coords (degrees) with scale factor
    P1 = np.array([
        [cd1[0, 0], cd1[0, 1], offset1[0]],
        [cd1[1, 0], cd1[1, 1], offset1[1]],
        [0, 0, deg2rad],
    ])

    # T: basis transformation between tangent planes (with projective scaling)
    T = np.array([
        [np.dot(ex1, ex2), np.dot(ey1, ex2), np.dot(n1, ex2) / deg2rad**2],
        [np.dot(ex1, ey2), np.dot(ey1, ey2), np.dot(n1, ey2) / deg2rad**2],
        [np.dot(ex1, n2) * deg2rad, np.dot(ey1, n2) * deg2rad, np.dot(n1, n2) / deg2rad],
    ])

    B = T @ P1

    # P2_inv: intermediate coords (degrees) -> pixel2
    cd2_inv = np.linalg.inv(cd2)
    P2_inv = np.array([
        [cd2_inv[0, 0], cd2_inv[0, 1], crpix2_0[0]],
        [cd2_inv[1, 0], cd2_inv[1, 1], crpix2_0[1]],
        [0, 0, 1],
    ])

    M = P2_inv @ B
    return M / M[2, 2]


def _tangent_rotation(wcs1, wcs2):
    """
    Rotation ``Q = B2^T B1`` taking gnomonic plane 1 to gnomonic plane 2.

    ``B_k = [ex_k, ey_k, n_k]`` (columns) is the orthonormal celestial-Cartesian
    basis of WCS ``k``. ``Q`` is the relative orientation of the two tangent
    frames; it is the identity when the two WCS share reference point and
    orientation.
    """
    n1, ex1, ey1 = get_tangent_basis(wcs1)
    n2, ex2, ey2 = get_tangent_basis(wcs2)
    B1 = np.column_stack([ex1, ey1, n1])
    B2 = np.column_stack([ex2, ey2, n2])
    return B2.T @ B1


class PlaneToPlaneTransform:
    """
    Precomputed exact plane-to-plane transform between two zenithal WCS.

    Construct with `compute_transform`; apply with `apply_transform` or the
    `apply` method. The per-pixel cost is one 2x2 affine, one radial rescale,
    one 3x3 multiply with a homogeneous divide, a second radial rescale, and a
    second 2x2 affine. For TAN -> TAN the whole map is a single 3x3 matrix,
    stored in ``matrix`` and used as a fast path.
    """

    def __init__(self, proj1, proj2, cd1, crpix1, cd2_inv, crpix2, rotation, matrix=None):
        self.proj1 = proj1
        self.proj2 = proj2
        self.cd1 = cd1
        self.crpix1 = crpix1
        self.cd2_inv = cd2_inv
        self.crpix2 = crpix2
        self.rotation = rotation
        self.matrix = matrix

    def apply(self, px1, py1, xp=None):
        return apply_transform(self, px1, py1, xp=xp)


def compute_transform(wcs1, wcs2):
    """
    Build the exact plane-to-plane transform between two zenithal WCS.

    Parameters
    ----------
    wcs1 : `~astropy.wcs.WCS`
        Source WCS.
    wcs2 : `~astropy.wcs.WCS`
        Target WCS.

    Returns
    -------
    transform : `PlaneToPlaneTransform`
        Precomputed transform, exact (to floating-point precision) for any pair
        of supported zenithal projections describing the same celestial frame.
    """
    proj1 = validate_wcs(wcs1, "wcs1")
    proj2 = validate_wcs(wcs2, "wcs2")
    validate_same_frame(wcs1, wcs2)

    cd1 = get_cd_matrix(wcs1)
    cd2_inv = np.linalg.inv(get_cd_matrix(wcs2))
    crpix1 = wcs1.wcs.crpix - 1  # 0-indexed
    crpix2 = wcs2.wcs.crpix - 1
    rotation = _tangent_rotation(wcs1, wcs2)

    # TAN -> TAN collapses to a single homography; precompute it as a fast path.
    matrix = None
    if proj1 == "TAN" and proj2 == "TAN":
        matrix = _compute_matrix_analytical(wcs1, wcs2)

    return PlaneToPlaneTransform(
        proj1, proj2, cd1, crpix1, cd2_inv, crpix2, rotation, matrix
    )


def compute_transform_matrix(wcs1, wcs2):
    """
    Return the single 3x3 projective matrix for a TAN -> TAN transform.

    Only TAN -> TAN reduces to a single matrix. For any other pair of zenithal
    projections the map includes radial rescales that are not projective; use
    `compute_transform` and `apply_transform` instead.
    """
    transform = compute_transform(wcs1, wcs2)
    if transform.matrix is None:
        raise ValueError(
            f"{transform.proj1} -> {transform.proj2} is not a single projective "
            "matrix; use compute_transform / apply_transform instead."
        )
    return transform.matrix


def _resolve_namespace(xp, *arrays):
    if xp is not None:
        return xp
    for arr in arrays:
        if hasattr(arr, "__array_namespace__"):
            return arr.__array_namespace__()
    return np


def _apply_matrix(matrix, px1, py1, xp):
    """Apply a single 3x3 projective matrix (TAN -> TAN fast path)."""
    m = np.asarray(matrix)
    denom = m[2, 0] * px1 + m[2, 1] * py1 + m[2, 2]
    px2 = (m[0, 0] * px1 + m[0, 1] * py1 + m[0, 2]) / denom
    py2 = (m[1, 0] * px1 + m[1, 1] * py1 + m[1, 2]) / denom
    return px2, py2


def _apply_general(transform, px1, py1, xp):
    """Apply the exact five-step zenithal plane-to-plane map."""
    cd1 = transform.cd1
    crpix1 = transform.crpix1
    cd2_inv = transform.cd2_inv
    crpix2 = transform.crpix2
    q = transform.rotation

    # 1. Pixel -> intermediate (degrees), input WCS.
    dx = px1 - float(crpix1[0])
    dy = py1 - float(crpix1[1])
    x = float(cd1[0, 0]) * dx + float(cd1[0, 1]) * dy
    y = float(cd1[1, 0]) * dx + float(cd1[1, 1]) * dy

    # 2. De-project to gnomonic (input projection), exactly.
    u1, v1, valid_in = deproject_to_gnomonic(x, y, transform.proj1, xp)

    # 3. Gnomonic homography, plane 1 -> plane 2.
    a = float(q[0, 0]) * u1 + float(q[0, 1]) * v1 + float(q[0, 2])
    b = float(q[1, 0]) * u1 + float(q[1, 1]) * v1 + float(q[1, 2])
    w = float(q[2, 0]) * u1 + float(q[2, 1]) * v1 + float(q[2, 2])
    visible = w > 0  # point must lie in the hemisphere plane 2 can image
    w_safe = xp.where(visible, w, xp.ones_like(w))
    u2 = a / w_safe
    v2 = b / w_safe

    # 4. Re-project from gnomonic (output projection), exactly.
    x2, y2 = reproject_from_gnomonic(u2, v2, transform.proj2, xp)

    # 5. Intermediate -> pixel (degrees), output WCS.
    px2 = float(crpix2[0]) + float(cd2_inv[0, 0]) * x2 + float(cd2_inv[0, 1]) * y2
    py2 = float(crpix2[1]) + float(cd2_inv[1, 0]) * x2 + float(cd2_inv[1, 1]) * y2

    valid = valid_in & visible
    nan = float("nan")
    px2 = xp.where(valid, px2, nan)
    py2 = xp.where(valid, py2, nan)
    return px2, py2


def apply_transform(transform, px1, py1, xp=None):
    """
    Apply a plane-to-plane transform to pixel coordinates.

    Parameters
    ----------
    transform : `PlaneToPlaneTransform`
        Transform from `compute_transform`.
    px1, py1 : array_like
        Source pixel coordinates (0-indexed).
    xp : module, optional
        Array namespace (numpy, jax.numpy, torch, cupy). If None, it is inferred
        from the inputs, defaulting to numpy.

    Returns
    -------
    px2, py2 : array
        Transformed pixel coordinates. Pixels with no valid mapping (out of the
        input projection domain, or beyond the output plane's horizon) are NaN.
    """
    xp = _resolve_namespace(xp, px1, py1)
    px1 = xp.asarray(px1)
    py1 = xp.asarray(py1)

    if transform.matrix is not None:
        return _apply_matrix(transform.matrix, px1, py1, xp)
    return _apply_general(transform, px1, py1, xp)
