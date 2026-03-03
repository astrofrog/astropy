"""
Transformation matrix computation and application.

This module computes the projective transformation matrix between two WCS
and provides functions to apply the transformation to pixel coordinates.
"""

import numpy as np

from ._projections import direction_to_intermediate, intermediate_to_direction
from ._wcs import get_cd_matrix, get_projection, get_tangent_basis, validate_wcs


def _solve_homography(src, dst):
    """Solve for 3x3 homography matrix using DLT algorithm."""
    num_points = src.shape[0]
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x, y = src[i]
        xp, yp = dst[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _compute_matrix_analytical(wcs1, wcs2):
    """
    Compute transformation matrix analytically (exact for TAN).

    This is a fully analytical computation using only WCS header parameters
    (CRVAL, CRPIX, CD/PC+CDELT). No WCS coordinate conversions are performed.
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


def _compute_matrix_zenithal(wcs1, wcs2):
    """
    Compute transformation matrix for zenithal projections analytically.

    Uses 3D direction vectors as the intermediate representation:
    pixel1 -> intermediate1 -> 3D direction -> intermediate2 -> pixel2

    This is exact for all zenithal projections (no fitting approximation
    in the projection math itself - only the final homography fit).
    """
    proj1 = get_projection(wcs1)
    proj2 = get_projection(wcs2)

    cd1 = get_cd_matrix(wcs1)
    cd2 = get_cd_matrix(wcs2)
    cd2_inv = np.linalg.inv(cd2)

    crpix1_0 = wcs1.wcs.crpix - 1  # 0-indexed
    crpix2_0 = wcs2.wcs.crpix - 1

    n1, ex1, ey1 = get_tangent_basis(wcs1)
    n2, ex2, ey2 = get_tangent_basis(wcs2)

    # Sample points to build the homography
    offsets = np.array([-100, 0, 100])
    px, py = np.meshgrid(crpix1_0[0] + offsets, crpix1_0[1] + offsets)
    src = np.column_stack([px.ravel(), py.ravel()])

    dst = []
    for pix1 in src:
        # Pixel1 -> intermediate1 (degrees)
        xy1 = cd1 @ (pix1 - crpix1_0)

        # Intermediate1 -> 3D direction
        d = intermediate_to_direction(xy1[0], xy1[1], proj1, n1, ex1, ey1)

        if d is None:
            # Point cannot be projected - skip
            dst.append([np.nan, np.nan])
            continue

        # 3D direction -> intermediate2 (degrees)
        x2, y2 = direction_to_intermediate(d, proj2, n2, ex2, ey2)

        # Intermediate2 -> pixel2
        pix2 = cd2_inv @ np.array([x2, y2]) + crpix2_0
        dst.append(pix2)

    dst = np.array(dst)

    # Remove any NaN points
    valid = ~np.isnan(dst[:, 0])
    if np.sum(valid) < 4:
        raise ValueError("Not enough valid calibration points")

    return _solve_homography(src[valid], dst[valid])


def compute_transform_matrix(wcs1, wcs2, method="auto"):
    """
    Compute the projective transformation matrix between two WCS.

    Parameters
    ----------
    wcs1 : `~astropy.wcs.WCS`
        Source WCS.
    wcs2 : `~astropy.wcs.WCS`
        Target WCS.
    method : str, optional
        'auto' (default): analytical for TAN-to-TAN, calibration otherwise
        'analytical': exact formula (TAN only)
        'calibration': fit to sample points (all zenithal)

    Returns
    -------
    M : ndarray
        3x3 projective transformation matrix.
    """
    proj1 = validate_wcs(wcs1, "wcs1")
    proj2 = validate_wcs(wcs2, "wcs2")

    if method == "auto":
        method = "analytical" if (proj1 == "TAN" and proj2 == "TAN") else "calibration"

    if method == "analytical":
        if proj1 != "TAN" or proj2 != "TAN":
            raise ValueError("Analytical method requires TAN projection for both WCS")
        return _compute_matrix_analytical(wcs1, wcs2)

    return _compute_matrix_zenithal(wcs1, wcs2)


def apply_transform(px1, py1, matrix, xp=None):
    """
    Apply projective transformation to pixel coordinates.

    Parameters
    ----------
    px1, py1 : array_like
        Source pixel coordinates.
    matrix : array_like
        3x3 transformation matrix from `compute_transform_matrix`.
    xp : module, optional
        Array namespace (numpy, jax.numpy, torch, cupy). If None, uses numpy
        or infers from input arrays.

    Returns
    -------
    px2, py2 : array
        Transformed pixel coordinates.
    """
    if xp is None:
        # Try to infer namespace from inputs
        for arr in (px1, py1, matrix):
            if hasattr(arr, "__array_namespace__"):
                xp = arr.__array_namespace__()
                break
        else:
            xp = np

    px1 = xp.asarray(px1)
    py1 = xp.asarray(py1)
    M = xp.asarray(matrix)

    denom = M[2, 0] * px1 + M[2, 1] * py1 + M[2, 2]
    px2 = (M[0, 0] * px1 + M[0, 1] * py1 + M[0, 2]) / denom
    py2 = (M[1, 0] * px1 + M[1, 1] * py1 + M[1, 2]) / denom

    return px2, py2
