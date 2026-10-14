# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy.testing import assert_allclose

from astropy.wcs import WCS
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
        "celestial": {"type": "space", "frame": "FK5", "equinox": "J2000.000"}
    }


def test_spectral_description():
    systems = spectral_cube_wcs(specsys="BARYCENT").world_axis_coordinate_systems
    assert systems["spectral"] == {
        "type": "spectral",
        "refposition": "BARYCENTER",
        "observer": None,
    }
    # SPECSYS values without a term in the IVOA refposition vocabulary
    # (pending a VEP) must not be described at all
    assert (
        "spectral"
        not in spectral_cube_wcs(specsys="LSRK").world_axis_coordinate_systems
    )
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
    assert world_axes_equivalent(
        spectral_cube_wcs(specsys="BARYCENT"), spectral_cube_wcs(specsys="BARYCENT")
    )
    assert not world_axes_equivalent(
        spectral_cube_wcs(specsys="LSRK"), spectral_cube_wcs(specsys="BARYCENT")
    )
    # Spectral systems that cannot be described with IVOA vocabulary terms
    # (pending a VEP for LSRK) must not match, even each other
    assert not world_axes_equivalent(
        spectral_cube_wcs(specsys="LSRK"), spectral_cube_wcs(specsys="LSRK")
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


def test_spectral_frame_to_coordinate_system():
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.wcs.wcsapi import spectral_frame_to_coordinate_system

    assert spectral_frame_to_coordinate_system("") is None

    # No IVOA refposition term yet (pending a VEP), so not describable
    assert spectral_frame_to_coordinate_system("LSRK") is None

    assert spectral_frame_to_coordinate_system("BARYCENT") == {
        "type": "spectral",
        "refposition": "BARYCENTER",
        "observer": None,
    }

    system = spectral_frame_to_coordinate_system(
        "TOPOCENT",
        observer_location=EarthLocation(1, 2, 3, unit=u.m),
        observer_time=Time(60000.0, format="mjd", scale="utc"),
        doppler_convention="radio",
        doppler_rest=1.420405751768 * u.GHz,
    )
    assert system == {
        "type": "spectral",
        "refposition": "TOPOCENTER",
        "observer": {"obsgeo_m": [1.0, 2.0, 3.0], "obstime_mjd": 60000.0},
        "doppler_convention": "radio",
        "doppler_rest_hz": 1.420405751768e9,
    }

    # Rest wavelengths and frequencies should give matching descriptions
    by_wavelength = spectral_frame_to_coordinate_system(
        "GEOCENTR", doppler_convention="optical", doppler_rest=0.21106114 * u.m
    )
    assert by_wavelength["refposition"] == "GEOCENTER"
    assert_allclose(by_wavelength["doppler_rest_hz"], 1.4204058e9, rtol=1e-6)


def test_time_frame_to_coordinate_system():
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.wcs.wcsapi import time_frame_to_coordinate_system

    assert time_frame_to_coordinate_system(Time(55197.0, format="mjd", scale="tt")) == {
        "type": "time",
        "timescale": "TT",
        "timeorigin_mjd": 55197.0,
        "location_m": None,
    }

    # The 'local' scale has no IVOA timescale term, so is not describable
    assert (
        time_frame_to_coordinate_system(Time(55197.0, format="mjd", scale="local"))
        is None
    )

    located = Time(
        55197.0, format="mjd", scale="utc", location=EarthLocation(1, 2, 3, unit=u.m)
    )
    assert time_frame_to_coordinate_system(located)["location_m"] == [1.0, 2.0, 3.0]
