"""
WCS header parsing utilities.

This module provides functions to extract and validate information from
WCS objects for use in the fast pixel-to-pixel transformation.
"""

import numpy as np

from ._projections import SUPPORTED_PROJECTIONS


def get_projection(wcs):
    """Extract projection code (e.g., 'TAN') from WCS CTYPE."""
    ctype = wcs.wcs.ctype[0]
    if len(ctype) >= 8 and ctype[4] == "-":
        return ctype[5:8]
    return None


def validate_wcs(wcs, name="wcs"):
    """Validate that WCS uses a supported zenithal projection."""
    proj = get_projection(wcs)
    if proj is None:
        raise ValueError(f"{name} has unrecognized CTYPE: {wcs.wcs.ctype}")
    if proj not in SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"{name} uses '{proj}' projection. Only zenithal projections "
            f"are supported: {', '.join(sorted(SUPPORTED_PROJECTIONS))}"
        )
    return proj


def get_cd_matrix(wcs):
    """Get the CD matrix from WCS, handling CDELT+PC or CD conventions."""
    if hasattr(wcs.wcs, "cd") and wcs.wcs.cd is not None:
        return wcs.wcs.cd.copy()
    return np.diag(wcs.wcs.get_cdelt()) @ wcs.wcs.get_pc()


def get_tangent_basis(wcs):
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
