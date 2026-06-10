# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Plain-data descriptions of world coordinate systems.

This module provides helpers for the
`~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_coordinate_systems` property:
converting astropy coordinate frames to plain-data descriptions using IVOA
vocabulary terms where possible, and comparing two WCSes to determine whether
their world coordinates are interchangeable at the values level.
"""

import numbers

__all__ = ["celestial_frame_to_coordinate_system", "world_axes_equivalent"]

# Mapping from FITS WCS Paper III SPECSYS values to reference position terms.
# These descend from the STC reference position vocabulary
# (http://www.ivoa.net/rdf/refposition); values without a blessed term are
# emitted with a "custom:" prefix by the FITS WCS implementation.
SPECSYS_TO_REFPOSITION = {
    "TOPOCENT": "TOPOCENTER",
    "GEOCENTR": "GEOCENTER",
    "BARYCENT": "BARYCENTER",
    "HELIOCEN": "HELIOCENTER",
    "LSRK": "LSRK",
    "LSRD": "LSRD",
    "GALACTOC": "GALACTIC_CENTER",
    "LOCALGRP": "LOCAL_GROUP_CENTER",
    "CMBDIPOL": "custom:CMBDIPOL",
    "SOURCE": "custom:SOURCE",
}


def celestial_frame_to_coordinate_system(frame):
    """
    Convert a celestial frame to a plain-data coordinate system description.

    The ``"frame"`` item uses terms from the IVOA reference frame vocabulary
    (http://www.ivoa.net/rdf/refframe), matching the ``system`` attribute of
    the VOTable ``COOSYS`` element.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame`
        The frame to describe.

    Returns
    -------
    dict or None
        A plain-data description of the frame, or `None` if the frame cannot
        be described with the known vocabulary (in which case it should be
        omitted from
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_coordinate_systems`).
    """
    from astropy.coordinates import FK4, FK5, ICRS, FK4NoETerms, Galactic, Supergalactic

    # Note that the frame *class* must match exactly (not a subclass), since
    # subclasses can change the meaning of the coordinates.
    if type(frame) is ICRS:
        return {"type": "space", "frame": "ICRS"}
    elif type(frame) is FK5:
        return {"type": "space", "frame": "eq_FK5", "equinox": frame.equinox.jyear_str}
    elif type(frame) is FK4:
        return {
            "type": "space",
            "frame": "eq_FK4",
            "equinox": frame.equinox.byear_str,
            "epoch": frame.obstime.byear_str,
        }
    elif type(frame) is FK4NoETerms:
        return {
            "type": "space",
            "frame": "custom:eq_FK4_no_e",
            "equinox": frame.equinox.byear_str,
            "epoch": frame.obstime.byear_str,
        }
    elif type(frame) is Galactic:
        return {"type": "space", "frame": "GALACTIC"}
    elif type(frame) is Supergalactic:
        return {"type": "space", "frame": "SUPER_GALACTIC"}
    else:
        return None


def _plain_data_equal(value1, value2):
    """
    Compare two values, returning `True` only if they are equal *and* are
    composed entirely of plain data types. Anything that is not plain data
    compares unequal, so that descriptions containing unexpected objects can
    never enable a fast path.
    """
    if isinstance(value1, dict) and isinstance(value2, dict):
        return value1.keys() == value2.keys() and all(
            _plain_data_equal(value1[key], value2[key]) for key in value1
        )
    elif isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
        return len(value1) == len(value2) and all(
            _plain_data_equal(item1, item2) for item1, item2 in zip(value1, value2)
        )
    elif isinstance(value1, (str, bool)) or isinstance(value2, (str, bool)):
        return type(value1) is type(value2) and value1 == value2
    elif value1 is None or value2 is None:
        return value1 is None and value2 is None
    elif isinstance(value1, numbers.Real) and isinstance(value2, numbers.Real):
        return value1 == value2
    else:
        return False


def world_axes_equivalent(wcs_in, wcs_out):
    """
    Whether world values from one WCS can be passed directly to another.

    Returns `True` only if it can be determined from the world axis metadata
    that the values returned by ``wcs_in.pixel_to_world_values`` carry the
    same meaning as the values accepted by ``wcs_out.world_to_pixel_values``
    axis-by-axis - that is, the high-level object round-trip between the two
    WCSes is the identity on raw world values. This requires both WCSes to
    describe their coordinate systems via
    `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_coordinate_systems`;
    absent or unrecognized descriptions result in `False` (which simply means
    equivalence could not be proven, not that the WCSes differ).

    Parameters
    ----------
    wcs_in, wcs_out : `~astropy.wcs.wcsapi.BaseLowLevelWCS` or `~astropy.wcs.wcsapi.BaseHighLevelWCS`
        The WCS transformations to compare.

    Returns
    -------
    bool
    """
    wcs_in = getattr(wcs_in, "low_level_wcs", wcs_in)
    wcs_out = getattr(wcs_out, "low_level_wcs", wcs_out)

    if wcs_in is wcs_out:
        return True

    if wcs_in.world_n_dim != wcs_out.world_n_dim:
        return False

    if list(wcs_in.world_axis_physical_types) != list(
        wcs_out.world_axis_physical_types
    ):
        return False

    if list(wcs_in.world_axis_units) != list(wcs_out.world_axis_units):
        return False

    systems_in = getattr(wcs_in, "world_axis_coordinate_systems", None)
    systems_out = getattr(wcs_out, "world_axis_coordinate_systems", None)

    if not systems_in or not systems_out:
        return False

    components_in = wcs_in.world_axis_object_components
    components_out = wcs_out.world_axis_object_components

    # Each world axis must map to the same component index of an equivalently
    # described coordinate system, and the grouping of axes into systems must
    # be consistent between the two WCSes (the key strings themselves are
    # arbitrary and do not need to match).
    key_map = {}
    for (key_in, index_in, _), (key_out, index_out, _) in zip(
        components_in, components_out
    ):
        if index_in != index_out:
            return False
        if key_map.setdefault(key_in, key_out) != key_out:
            return False
        system_in = systems_in.get(key_in)
        system_out = systems_out.get(key_out)
        if not system_in or not system_out:
            return False
        if not _plain_data_equal(system_in, system_out):
            return False

    if len(set(key_map.values())) != len(key_map):
        return False

    return True
