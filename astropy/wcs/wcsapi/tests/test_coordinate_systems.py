# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from astropy.wcs.wcsapi import world_axes_equivalent
from astropy.wcs.wcsapi.wrappers import SlicedLowLevelWCS


def celestial_wcs(ctype1="RA---TAN", ctype2="DEC--TAN", radesys=None, equinox=None):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [ctype1, ctype2]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crval = [30.0, 40.0]
    wcs.wcs.crpix = [10.0, 20.0]
    wcs.wcs.cdelt = [-0.01, 0.01]
    if radesys is not None:
        wcs.wcs.radesys = radesys
    if equinox is not None:
        wcs.wcs.equinox = equinox
    return wcs


def spectral_cube_wcs(specsys="LSRK", restfrq=0.0):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ"]
    wcs.wcs.cunit = ["deg", "deg", "Hz"]
    wcs.wcs.crval = [30.0, 40.0, 1.4e9]
    wcs.wcs.crpix = [10.0, 20.0, 30.0]
    wcs.wcs.cdelt = [-0.01, 0.01, 1e6]
    wcs.wcs.specsys = specsys
    wcs.wcs.restfrq = restfrq
    return wcs


def test_celestial_descriptions():
    assert celestial_wcs().world_axis_coordinate_systems == {
        "celestial": {"type": "space", "frame": "ICRS"}
    }
    assert celestial_wcs(
        radesys="FK5", equinox=2000.0
    ).world_axis_coordinate_systems == {
        "celestial": {"type": "space", "frame": "eq_FK5", "equinox": "J2000.000"}
    }


def test_spectral_description():
    systems = spectral_cube_wcs().world_axis_coordinate_systems
    assert systems["spectral"] == {
        "type": "spectral",
        "refposition": "LSRK",
        "observer": None,
    }
    # Without SPECSYS the spectral system is unknown and must not be described
    assert "spectral" not in spectral_cube_wcs(specsys="").world_axis_coordinate_systems


def test_generic_and_stokes_descriptions():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["", "STOKES"]
    systems = wcs.world_axis_coordinate_systems
    assert systems["world"] == {"type": "generic"}
    assert systems["stokes"] == {"type": "stokes"}


def test_equivalent_same_frame_different_projection():
    wcs1 = celestial_wcs()
    wcs2 = celestial_wcs(ctype1="RA---SIN", ctype2="DEC--SIN")
    assert world_axes_equivalent(wcs1, wcs2)
    assert world_axes_equivalent(wcs1, wcs1)


def test_not_equivalent_different_frame():
    icrs = celestial_wcs()
    fk5 = celestial_wcs(radesys="FK5", equinox=2000.0)
    fk5_1950 = celestial_wcs(radesys="FK5", equinox=1950.0)
    galactic = celestial_wcs(ctype1="GLON-TAN", ctype2="GLAT-TAN")
    assert not world_axes_equivalent(icrs, fk5)
    assert not world_axes_equivalent(fk5, fk5_1950)
    assert not world_axes_equivalent(icrs, galactic)


def test_not_equivalent_swapped_axes():
    wcs1 = celestial_wcs()
    wcs2 = celestial_wcs(ctype1="DEC--TAN", ctype2="RA---TAN")
    assert not world_axes_equivalent(wcs1, wcs2)


def test_spectral_equivalence():
    assert world_axes_equivalent(spectral_cube_wcs(), spectral_cube_wcs())
    assert not world_axes_equivalent(
        spectral_cube_wcs(specsys="LSRK"), spectral_cube_wcs(specsys="BARYCENT")
    )
    # Unknown spectral systems must never match, even each other
    assert not world_axes_equivalent(
        spectral_cube_wcs(specsys=""), spectral_cube_wcs(specsys="")
    )


def test_sliced_wcs_preserves_descriptions():
    wcs = spectral_cube_wcs()
    sliced = SlicedLowLevelWCS(wcs, 0)  # drop the spectral axis
    assert sliced.world_axis_coordinate_systems == {
        "celestial": {"type": "space", "frame": "ICRS"}
    }
    assert world_axes_equivalent(sliced, celestial_wcs())


def test_pixel_to_pixel_fast_path_results():
    wcs1 = celestial_wcs()
    wcs2 = celestial_wcs(ctype1="RA---SIN", ctype2="DEC--SIN")

    x = np.array([1.0, 5.0, 30.0])
    y = np.array([2.0, 7.0, 40.0])

    fast = pixel_to_pixel(wcs1, wcs2, x, y)

    # Compute the reference result through the high-level objects
    world = wcs1.pixel_to_world(x, y)
    slow = wcs2.world_to_pixel(world)

    assert_allclose(fast[0], slow[0])
    assert_allclose(fast[1], slow[1])


def test_pixel_to_pixel_fast_path_taken():
    wcs1 = celestial_wcs()
    wcs2 = celestial_wcs(ctype1="RA---SIN", ctype2="DEC--SIN")

    def fail(*args, **kwargs):
        raise AssertionError("high-level path should not be used")

    wcs1.pixel_to_world = fail
    wcs2.world_to_pixel = fail

    result = pixel_to_pixel(wcs1, wcs2, np.array([1.0, 5.0]), np.array([2.0, 7.0]))
    assert len(result) == 2


def test_pixel_to_pixel_slow_path_taken():
    wcs1 = celestial_wcs()
    wcs2 = celestial_wcs(ctype1="GLON-TAN", ctype2="GLAT-TAN")

    fast = pixel_to_pixel(wcs1, wcs2, np.array([1.0, 5.0]), np.array([2.0, 7.0]))

    world = wcs1.pixel_to_world(np.array([1.0, 5.0]), np.array([2.0, 7.0]))
    slow = wcs2.world_to_pixel(world)

    assert_allclose(fast[0], slow[0])
    assert_allclose(fast[1], slow[1])


class UndescribedWCS(SlicedLowLevelWCS):
    # Simulates a low-level WCS that does not override the new property and
    # therefore asserts nothing about its coordinate systems.
    @property
    def world_axis_coordinate_systems(self):
        return {}


def test_unknown_never_matches():
    wcs = celestial_wcs()
    undescribed1 = UndescribedWCS(wcs, [slice(None), slice(None)])
    undescribed2 = UndescribedWCS(wcs, [slice(None), slice(None)])
    assert not world_axes_equivalent(undescribed1, wcs)
    assert not world_axes_equivalent(undescribed1, undescribed2)
