"""
Zenithal projection formulas for coordinate transformations.

This module implements the radial rescales that convert any supported zenithal
(azimuthal) projection to and from the gnomonic (``TAN``) projection, as defined
in the FITS WCS standard (Calabretta & Greisen 2002). The conversion is exact
(to floating-point precision): a zenithal projection depends on the angular
distance from the reference point only through a monotonic radial law, so the
gnomonic radius is recoverable in closed form and no reconstruction of the full
spherical coordinates is needed.

For a zenithal projection the intermediate world coordinates ``(x, y)`` (degrees)
satisfy ``x = R sin(phi)``, ``y = -R cos(phi)`` with ``R`` the projection radius.
Writing the dimensionless radius ``psi = (pi/180) R`` and the gnomonic radius
``G = tan(rho)`` (with ``rho`` the angular distance from the reference point), the
de-projection to gnomonic multiplies ``(x, y)`` by ``f_down = G / psi`` and the
re-projection from gnomonic multiplies the gnomonic coordinates by
``f_up = psi / G``. Both factors are 1 at the reference point and identically 1
for ``TAN``; their closed forms are implemented below.

All functions here are elementwise over arrays and namespace-agnostic: they take
the array namespace ``xp`` (numpy, jax.numpy, torch, cupy, ...) so the same code
runs on CPU and accelerators.
"""

import numpy as np

DEG2RAD = np.pi / 180.0

# Zenithal projections supported by the plane-to-plane algorithm
SUPPORTED_PROJECTIONS = frozenset({
    "TAN",  # Gnomonic
    "SIN",  # Orthographic
    "STG",  # Stereographic
    "ARC",  # Zenithal equidistant
    "ZEA",  # Zenithal equal-area
})

# Upper bound on the dimensionless radius ``psi`` for which a point still lies in
# the ``rho < 90 deg`` hemisphere that the gnomonic intermediate can represent.
# Points at or beyond the bound have no gnomonic preimage and are flagged
# invalid (this also covers the SIN psi>1 / ZEA psi>2 "no sky preimage" cases,
# since those bounds are looser than the rho<90 bound used here).
_PSI_MAX = {
    "TAN": np.inf,
    "SIN": 1.0,
    "ARC": np.pi / 2.0,
    "STG": 2.0,
    "ZEA": np.sqrt(2.0),
}


def deproject_to_gnomonic(x_deg, y_deg, proj, xp):
    """
    De-project zenithal intermediate coordinates to gnomonic, exactly.

    Parameters
    ----------
    x_deg, y_deg : array_like
        Intermediate world coordinates in degrees for projection ``proj``.
    proj : str
        Projection code (TAN, SIN, STG, ARC, ZEA).
    xp : module
        Array namespace (numpy, jax.numpy, torch, cupy, ...).

    Returns
    -------
    u, v : array
        Dimensionless gnomonic coordinates,
        ``u = tan(rho) sin(phi)``, ``v = -tan(rho) cos(phi)``.
    valid : array of bool
        False where the pixel has no preimage in the ``rho < 90 deg``
        hemisphere (out-of-domain SIN/ZEA, or beyond the gnomonic horizon).
    """
    xr = x_deg * DEG2RAD
    yr = y_deg * DEG2RAD
    psi = xp.sqrt(xr * xr + yr * yr)
    valid = psi < _PSI_MAX[proj]

    # Clamp out-of-domain and zero radii so the closed-form factors stay finite
    # and warning-free. The invalid mask is returned for the caller to apply;
    # the zero radius is exact by continuity (f_down -> 1).
    psi = xp.where(valid, psi, 0.0)
    psi2 = psi * psi

    if proj == "TAN":
        f = 1.0
    elif proj == "SIN":
        f = 1.0 / xp.sqrt(1.0 - psi2)
    elif proj == "ARC":
        tiny = psi < 1e-12
        psi_safe = xp.where(tiny, 1.0, psi)
        f = xp.where(tiny, 1.0, xp.tan(psi) / psi_safe)
    elif proj == "STG":
        f = 1.0 / (1.0 - psi2 / 4.0)
    elif proj == "ZEA":
        f = xp.sqrt(1.0 - psi2 / 4.0) / (1.0 - psi2 / 2.0)
    else:
        raise ValueError(f"Unsupported projection: {proj}")

    return f * xr, f * yr, valid


def reproject_from_gnomonic(u, v, proj, xp):
    """
    Re-project gnomonic coordinates to zenithal intermediate coordinates, exactly.

    Parameters
    ----------
    u, v : array_like
        Dimensionless gnomonic coordinates.
    proj : str
        Projection code (TAN, SIN, STG, ARC, ZEA).
    xp : module
        Array namespace (numpy, jax.numpy, torch, cupy, ...).

    Returns
    -------
    x_deg, y_deg : array
        Intermediate world coordinates in degrees for projection ``proj``.

    Notes
    -----
    The gnomonic intermediate always satisfies ``rho < 90 deg``, so the output
    re-projection is never out of domain for any supported projection.
    """
    G2 = u * u + v * v

    if proj == "TAN":
        f = 1.0
    elif proj == "SIN":
        f = 1.0 / xp.sqrt(1.0 + G2)
    elif proj == "ARC":
        G = xp.sqrt(G2)
        tiny = G < 1e-12
        G_safe = xp.where(tiny, 1.0, G)
        f = xp.where(tiny, 1.0, xp.arctan(G) / G_safe)
    elif proj == "STG":
        f = 2.0 / (1.0 + xp.sqrt(1.0 + G2))
    elif proj == "ZEA":
        # Cancellation-free form of sqrt(2(1 - 1/sqrt(1+G^2)))/G:
        # with s = sqrt(1+G^2), 1 - 1/s = G^2/(s(s+1)), so the G divides out.
        # Finite at G=0 (-> 1), no small-radius guard needed.
        s = xp.sqrt(1.0 + G2)
        f = xp.sqrt(2.0 / (s * (1.0 + s)))
    else:
        raise ValueError(f"Unsupported projection: {proj}")

    return (f * u) / DEG2RAD, (f * v) / DEG2RAD
