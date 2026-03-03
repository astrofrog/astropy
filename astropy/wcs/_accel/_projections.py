"""
Zenithal projection formulas for coordinate transformations.

This module implements the forward and inverse projection equations for
zenithal (azimuthal) projections as defined in the FITS WCS standard
(Calabretta & Greisen 2002).
"""

import numpy as np

# Zenithal projections supported by the plane-to-plane algorithm
SUPPORTED_PROJECTIONS = frozenset({
    "TAN",  # Gnomonic
    "SIN",  # Orthographic
    "STG",  # Stereographic
    "ARC",  # Zenithal equidistant
    "ZEA",  # Zenithal equal-area
})


def intermediate_to_direction(x_deg, y_deg, proj, n, ex, ey):
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
    d : ndarray or None
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


def direction_to_intermediate(d, proj, n, ex, ey):
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
