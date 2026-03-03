"""
Fast pixel-to-pixel transformations for zenithal WCS projections.

This module provides accelerated coordinate transformations between two WCS
with zenithal projections (TAN, SIN, STG, ARC, ZEA). The transformation is
computed entirely from WCS header parameters without calling WCS coordinate
conversion methods.

The transformation is a projective (homographic) mapping:
    px2 = (A*px1 + B*py1 + C) / (G*px1 + H*py1 + I)
    py2 = (D*px1 + E*py1 + F) / (G*px1 + H*py1 + I)

This avoids per-pixel trigonometry, achieving significant speedups over
the standard pixel->world->pixel approach.
"""

import numpy as np

__all__ = [
    "compute_transform_matrix",
    "apply_transform",
    "apply_transform_grid",
    "SUPPORTED_PROJECTIONS",
]

# Zenithal projections supported by the plane-to-plane algorithm
SUPPORTED_PROJECTIONS = frozenset({
    "TAN",  # Gnomonic
    "SIN",  # Orthographic
    "STG",  # Stereographic
    "ARC",  # Zenithal equidistant
    "ZEA",  # Zenithal equal-area
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

    Returns (n, ex, ey) where:
    - n: unit vector pointing to CRVAL (reference direction)
    - ex: unit vector in tangent plane pointing east (increasing RA)
    - ey: unit vector in tangent plane pointing north (increasing Dec)
    """
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


def _intermediate_to_direction(x_deg, y_deg, proj, n, ex, ey):
    """
    Convert intermediate world coordinates to 3D unit direction vector.

    Parameters
    ----------
    x_deg, y_deg : float
        Intermediate world coordinates in degrees.
    proj : str
        Projection code (TAN, SIN, STG, ARC, ZEA).
    n, ex, ey : ndarray
        Tangent plane basis vectors.

    Returns
    -------
    d : ndarray
        Unit 3D direction vector, or None if point cannot be projected.
    """
    x = np.radians(x_deg)
    y = np.radians(y_deg)
    r = np.sqrt(x * x + y * y)

    if r < 1e-12:
        return n.copy()

    # Native spherical angle phi (from -y toward +x)
    phi = np.arctan2(x, -y)

    # Compute native latitude theta from radial distance r
    # In FITS WCS zenithal projections: theta = 90° at reference point
    # r = R(theta) where R is the projection's radial function
    if proj == "TAN":
        # r = cot(theta), so theta = arccot(r) = pi/2 - arctan(r)
        theta = np.pi / 2 - np.arctan(r)
    elif proj == "SIN":
        # r = cos(theta), so theta = arccos(r)
        if r > 1:
            return None
        theta = np.arccos(r)
    elif proj == "STG":
        # r = 2*tan((pi/2 - theta)/2), so theta = pi/2 - 2*arctan(r/2)
        theta = np.pi / 2 - 2 * np.arctan(r / 2)
    elif proj == "ARC":
        # r = pi/2 - theta (radians), so theta = pi/2 - r
        theta = np.pi / 2 - r
    elif proj == "ZEA":
        # r = 2*sin((pi/2 - theta)/2) = sqrt(2*(1 - sin(theta)))
        # Inverse: theta = pi/2 - 2*arcsin(r/2)
        sin_half = r / 2
        if sin_half > 1:
            return None
        theta = np.pi / 2 - 2 * np.arcsin(sin_half)
    else:
        return None

    if theta < 0:
        return None

    # 3D direction vector
    # theta is native latitude: 90° at pole (n), 0° at horizon
    # sin(theta) is component along n, cos(theta) is component in tangent plane
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Direction in tangent plane at angle phi from -ey
    tangent_dir = np.sin(phi) * ex - np.cos(phi) * ey

    d = sin_theta * n + cos_theta * tangent_dir
    return d / np.linalg.norm(d)


def _direction_to_intermediate(d, proj, n, ex, ey):
    """
    Convert 3D unit direction vector to intermediate world coordinates.

    Parameters
    ----------
    d : ndarray
        Unit 3D direction vector.
    proj : str
        Projection code (TAN, SIN, STG, ARC, ZEA).
    n, ex, ey : ndarray
        Tangent plane basis vectors.

    Returns
    -------
    x_deg, y_deg : float
        Intermediate world coordinates in degrees.
    """
    # Project onto tangent plane basis
    # sin_theta is component along n (pole direction)
    # cos_theta is component in tangent plane
    sin_theta = np.dot(d, n)
    dx = np.dot(d, ex)
    dy = np.dot(d, ey)
    cos_theta = np.sqrt(dx * dx + dy * dy)

    if cos_theta < 1e-12:
        return 0.0, 0.0

    # Native spherical angle phi (azimuth from -ey toward +ex)
    phi = np.arctan2(dx, -dy)

    # Native latitude theta (90° at pole, 0° at horizon)
    theta = np.arctan2(sin_theta, cos_theta)

    # Radial distance r = R(theta) (projection-specific)
    if proj == "TAN":
        # r = cot(theta) = cos(theta)/sin(theta)
        if sin_theta < 1e-12:
            return np.nan, np.nan
        r = cos_theta / sin_theta
    elif proj == "SIN":
        # r = cos(theta)
        r = cos_theta
    elif proj == "STG":
        # r = 2*tan((pi/2 - theta)/2) = 2*(1 - sin(theta))/cos(theta)
        if cos_theta < 1e-12:
            return np.nan, np.nan
        r = 2 * (1 - sin_theta) / cos_theta
    elif proj == "ARC":
        # r = pi/2 - theta
        r = np.pi / 2 - theta
    elif proj == "ZEA":
        # r = 2*sin((pi/2 - theta)/2) = sqrt(2*(1 - sin(theta)))
        r = np.sqrt(2 * (1 - sin_theta))
    else:
        return np.nan, np.nan

    # Intermediate coordinates: x = r*sin(phi), y = -r*cos(phi)
    x = r * np.sin(phi)
    y = -r * np.cos(phi)

    return np.degrees(x), np.degrees(y)


def _compute_matrix_zenithal(wcs1, wcs2):
    """
    Compute transformation matrix for zenithal projections analytically.

    Uses 3D direction vectors as the intermediate representation:
    pixel1 -> intermediate1 -> 3D direction -> intermediate2 -> pixel2

    This is exact for all zenithal projections (no fitting approximation
    in the projection math itself - only the final homography fit).
    """
    proj1 = _get_projection(wcs1)
    proj2 = _get_projection(wcs2)

    cd1 = _get_cd_matrix(wcs1)
    cd2 = _get_cd_matrix(wcs2)
    cd2_inv = np.linalg.inv(cd2)

    crpix1_0 = wcs1.wcs.crpix - 1  # 0-indexed
    crpix2_0 = wcs2.wcs.crpix - 1

    n1, ex1, ey1 = _get_tangent_basis(wcs1)
    n2, ex2, ey2 = _get_tangent_basis(wcs2)

    # Sample points to build the homography
    offsets = np.array([-100, 0, 100])
    px, py = np.meshgrid(crpix1_0[0] + offsets, crpix1_0[1] + offsets)
    src = np.column_stack([px.ravel(), py.ravel()])

    dst = []
    for pix1 in src:
        # Pixel1 -> intermediate1 (degrees)
        xy1 = cd1 @ (pix1 - crpix1_0)

        # Intermediate1 -> 3D direction
        d = _intermediate_to_direction(xy1[0], xy1[1], proj1, n1, ex1, ey1)

        if d is None:
            # Point cannot be projected - skip
            dst.append([np.nan, np.nan])
            continue

        # 3D direction -> intermediate2 (degrees)
        x2, y2 = _direction_to_intermediate(d, proj2, n2, ex2, ey2)

        # Intermediate2 -> pixel2
        pix2 = cd2_inv @ np.array([x2, y2]) + crpix2_0
        dst.append(pix2)

    dst = np.array(dst)

    # Remove any NaN points
    valid = ~np.isnan(dst[:, 0])
    if np.sum(valid) < 4:
        raise ValueError("Not enough valid calibration points")

    return _solve_homography(src[valid], dst[valid])


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
