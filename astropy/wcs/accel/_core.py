"""
Fast pixel-to-pixel transformations for zenithal WCS projections.

This module implements the Montage "plane-to-plane" algorithm for fast
coordinate transformations between two WCS with zenithal projections.
For TAN projections, the transformation is exact and computed entirely
from WCS header parameters (CRVAL, CRPIX, CD/PC+CDELT) without any
WCS coordinate conversions. For other zenithal projections (SIN, STG,
ARC, etc.), a highly accurate homography approximation is used.

The transformation is a projective (homographic) mapping:
    px2 = (A*px1 + B*py1 + C) / (G*px1 + H*py1 + I)
    py2 = (D*px1 + E*py1 + F) / (G*px1 + H*py1 + I)

This avoids per-pixel trigonometry, achieving 20-400x speedups over
the standard pixel->world->pixel approach.
"""

import numpy as np

__all__ = [
    "compute_transform_matrix",
    "apply_transform",
    "apply_transform_grid",
    "SUPPORTED_PROJECTIONS",
]

# Zenithal projections where the plane-to-plane algorithm works
SUPPORTED_PROJECTIONS = frozenset({
    "TAN",  # Gnomonic
    "SIN",  # Orthographic
    "STG",  # Stereographic
    "ARC",  # Zenithal equidistant
    "ZEA",  # Zenithal equal-area
    "ZPN",  # Zenithal polynomial
    "AZP",  # Zenithal perspective
    "AIR",  # Airy
    "NCP",  # North celestial pole (deprecated SIN variant)
})


def _get_projection(wcs):
    """Extract projection code (e.g., 'TAN') from WCS CTYPE."""
    ctype = wcs.wcs.ctype[0]
    if len(ctype) >= 8 and ctype[4] == "-":
        return ctype[5:8]
    return None


def _validate_wcs(wcs, name="wcs"):
    """Validate that WCS uses a supported zenithal projection."""
    proj = _get_projection(wcs)
    if proj is None:
        raise ValueError(f"{name} has unrecognized CTYPE: {wcs.wcs.ctype}")
    if proj not in SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"{name} uses '{proj}' projection. Only zenithal projections "
            f"are supported: {', '.join(sorted(SUPPORTED_PROJECTIONS))}"
        )
    return proj


def _get_cd_matrix(wcs):
    """Get the CD matrix from WCS, handling CDELT+PC or CD conventions."""
    if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
        return wcs.wcs.cd.copy()
    return np.diag(wcs.wcs.get_cdelt()) @ wcs.wcs.get_pc()


def _get_tangent_basis(wcs):
    """
    Compute tangent plane basis vectors analytically from CRVAL.

    For TAN projections, the tangent plane basis is determined entirely by
    the reference point (CRVAL). No WCS coordinate conversions are needed.

    Returns (n, ex, ey) where:
    - n: unit vector pointing to CRVAL
    - ex: unit vector in tangent plane pointing east (increasing RA)
    - ey: unit vector in tangent plane pointing north (increasing Dec)
    """
    # Get reference point from CRVAL
    crval = wcs.wcs.crval  # [RA, Dec] in degrees
    ra0 = np.radians(crval[0])
    dec0 = np.radians(crval[1])

    cos_dec = np.cos(dec0)
    sin_dec = np.sin(dec0)
    cos_ra = np.cos(ra0)
    sin_ra = np.sin(ra0)

    # Unit vector pointing to reference point
    n = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])

    # ex points east (direction of increasing RA at reference point)
    # In FITS WCS with standard LONPOLE=180, intermediate x increases eastward
    ex = np.array([-sin_ra, cos_ra, 0.0])

    # ey points north (direction of increasing Dec at reference point)
    ey = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])

    return n, ex, ey


def _compute_matrix_analytical(wcs1, wcs2):
    """
    Compute transformation matrix analytically (exact for TAN).

    This is a fully analytical computation using only WCS header parameters
    (CRVAL, CRPIX, CD/PC+CDELT). No WCS coordinate conversions are performed.
    """
    deg2rad = np.pi / 180.0

    cd1 = _get_cd_matrix(wcs1)
    cd2 = _get_cd_matrix(wcs2)

    # Convert CRPIX from FITS 1-indexed to 0-indexed pixel coordinates
    crpix1_0 = wcs1.wcs.crpix - 1
    crpix2_0 = wcs2.wcs.crpix - 1

    n1, ex1, ey1 = _get_tangent_basis(wcs1)
    n2, ex2, ey2 = _get_tangent_basis(wcs2)

    # Intermediate world coords at pixel (0,0): CD @ (0 - crpix_0) = -CD @ crpix_0
    offset1 = -cd1 @ crpix1_0

    # Matrix: pixel1 -> (u1, v1, deg2rad)
    P1 = np.array([
        [cd1[0, 0], cd1[0, 1], offset1[0]],
        [cd1[1, 0], cd1[1, 1], offset1[1]],
        [0, 0, deg2rad],
    ])

    # Matrix: (u1, v1, deg2rad) -> (dir.ex2/deg2rad, dir.ey2/deg2rad, dir.n2)
    T = np.array([
        [np.dot(ex1, ex2), np.dot(ey1, ex2), np.dot(n1, ex2) / deg2rad**2],
        [np.dot(ex1, ey2), np.dot(ey1, ey2), np.dot(n1, ey2) / deg2rad**2],
        [np.dot(ex1, n2) * deg2rad, np.dot(ey1, n2) * deg2rad, np.dot(n1, n2) / deg2rad],
    ])

    B = T @ P1

    # Matrix: (u2, v2, 1) -> pixel2 (0-indexed)
    cd2_inv = np.linalg.inv(cd2)
    P2_inv = np.array([
        [cd2_inv[0, 0], cd2_inv[0, 1], crpix2_0[0]],
        [cd2_inv[1, 0], cd2_inv[1, 1], crpix2_0[1]],
        [0, 0, 1],
    ])

    M = P2_inv @ B
    return M / M[2, 2]


def _solve_homography(src, dst):
    """Solve for 3x3 homography matrix using DLT algorithm."""
    n = src.shape[0]
    A = np.zeros((2 * n, 9))

    for i in range(n):
        x, y = src[i]
        xp, yp = dst[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _compute_matrix_calibration(wcs1, wcs2):
    """Compute transformation matrix by fitting to calibration points."""
    crpix = wcs1.wcs.crpix
    offsets = np.array([-100, 0, 100])
    px, py = np.meshgrid(crpix[0] + offsets, crpix[1] + offsets)
    src = np.column_stack([px.ravel(), py.ravel()])

    world = wcs1.pixel_to_world_values(src[:, 0], src[:, 1])
    px2, py2 = wcs2.world_to_pixel_values(*world)
    dst = np.column_stack([px2, py2])

    return _solve_homography(src, dst)


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
    proj1 = _validate_wcs(wcs1, "wcs1")
    proj2 = _validate_wcs(wcs2, "wcs2")

    if method == "auto":
        method = "analytical" if (proj1 == "TAN" and proj2 == "TAN") else "calibration"

    if method == "analytical":
        if proj1 != "TAN" or proj2 != "TAN":
            raise ValueError("Analytical method requires TAN projection for both WCS")
        return _compute_matrix_analytical(wcs1, wcs2)

    return _compute_matrix_calibration(wcs1, wcs2)


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


def apply_transform_grid(matrix, x_range, y_range, shape, xp=None):
    """
    Apply projective transformation to a regular pixel grid.

    Parameters
    ----------
    matrix : array_like
        3x3 transformation matrix from `compute_transform_matrix`.
    x_range : tuple
        (x_start, x_stop) pixel range.
    y_range : tuple
        (y_start, y_stop) pixel range.
    shape : tuple
        (ny, nx) output shape.
    xp : module, optional
        Array namespace (numpy, jax.numpy, torch, cupy). Defaults to numpy.

    Returns
    -------
    px2, py2 : array
        2D arrays of transformed coordinates with given shape.
    """
    if xp is None:
        xp = np

    ny, nx = shape
    M = xp.asarray(matrix)

    x = xp.linspace(float(x_range[0]), float(x_range[1]), nx)
    y = xp.linspace(float(y_range[0]), float(y_range[1]), ny)

    if hasattr(xp, "meshgrid"):
        px1, py1 = xp.meshgrid(x, y, indexing="xy")
    else:
        # Fallback for libraries without meshgrid
        px1 = xp.broadcast_to(x[None, :], (ny, nx))
        py1 = xp.broadcast_to(y[:, None], (ny, nx))

    denom = M[2, 0] * px1 + M[2, 1] * py1 + M[2, 2]
    px2 = (M[0, 0] * px1 + M[0, 1] * py1 + M[0, 2]) / denom
    py2 = (M[1, 0] * px1 + M[1, 1] * py1 + M[1, 2]) / denom

    return px2, py2
