from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseLowLevelWCS
from astropy.wcs.wcsapi.low_level_api import _WorldAxisClass, _WorldAxisComponent


@pytest.fixture
def spectral_1d_fitswcs() -> WCS:
    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ("FREQ",)
    wcs.wcs.cunit = ("Hz",)
    wcs.wcs.cdelt = (3.0e9,)
    wcs.wcs.crval = (4.0e9,)
    wcs.wcs.crpix = (11.0,)
    wcs.wcs.cname = ("Frequency",)
    return wcs


@pytest.fixture
def time_1d_fitswcs() -> WCS:
    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ("TIME",)
    wcs.wcs.mjdref = (30042, 0)
    wcs.wcs.crval = (3.0,)
    wcs.wcs.crpix = (11.0,)
    wcs.wcs.cname = ("Time",)
    wcs.wcs.cunit = "s"
    return wcs


@pytest.fixture
def celestial_2d_fitswcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---CAR", "DEC--CAR"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = -2.0, 2.0
    wcs.wcs.crval = 4.0, 0.0
    wcs.wcs.crpix = 6.0, 7.0
    wcs.wcs.cname = "Right Ascension", "Declination"
    wcs.pixel_shape = (6, 7)
    wcs.pixel_bounds = [(-1, 5), (1, 7)]
    return wcs


@pytest.fixture
def spectral_cube_3d_fitswcs() -> WCS:
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = "RA---CAR", "DEC--CAR", "FREQ"
    wcs.wcs.cunit = "deg", "deg", "Hz"
    wcs.wcs.cdelt = -2.0, 2.0, 3.0e9
    wcs.wcs.crval = 4.0, 0.0, 4.0e9
    wcs.wcs.crpix = 6.0, 7.0, 11.0
    wcs.wcs.cname = "Right Ascension", "Declination", "Frequency"
    wcs.pixel_shape = (6, 7, 3)
    wcs.pixel_bounds = [(-1, 5), (1, 7), (1, 2.5)]
    return wcs


@pytest.fixture
def cube_4d_fitswcs() -> WCS:
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = "RA---CAR", "DEC--CAR", "FREQ", "TIME"
    wcs.wcs.cunit = "deg", "deg", "Hz", "s"
    wcs.wcs.cdelt = -2.0, 2.0, 3.0e9, 1
    wcs.wcs.crval = 4.0, 0.0, 4.0e9, 3
    wcs.wcs.crpix = 6.0, 7.0, 11.0, 11.0
    wcs.wcs.cname = "Right Ascension", "Declination", "Frequency", "Time"
    wcs.wcs.mjdref = (30042, 0)
    return wcs


class Spectral1DLowLevelWCS(BaseLowLevelWCS):
    @property
    def pixel_n_dim(self) -> int:
        return 1

    @property
    def world_n_dim(self) -> int:
        return 1

    @property
    def world_axis_physical_types(self) -> Sequence[str | None]:
        return ("em.freq",)

    @property
    def world_axis_units(self) -> Sequence[str]:
        return ("Hz",)

    @property
    def world_axis_names(self) -> Sequence[str]:
        return ("Frequency",)

    _pixel_shape: tuple[int, ...] | None = None

    @property
    def pixel_shape(self) -> tuple[int, ...] | None:
        return self._pixel_shape

    @pixel_shape.setter
    def pixel_shape(self, value: tuple[int, ...] | None) -> None:
        self._pixel_shape = value

    _pixel_bounds: Sequence[tuple[float, float] | None] | None = None

    @property
    def pixel_bounds(self) -> Sequence[tuple[float, float] | None] | None:
        return self._pixel_bounds

    @pixel_bounds.setter
    def pixel_bounds(self, value: Sequence[tuple[float, float] | None] | None) -> None:
        self._pixel_bounds = value

    def pixel_to_world_values(self, *pixel_arrays: ArrayLike) -> Any:
        (pixel_array,) = pixel_arrays
        return (np.asarray(pixel_array) - 10) * 3e9 + 4e9

    def world_to_pixel_values(self, *world_arrays: ArrayLike) -> Any:
        (world_array,) = world_arrays
        return (np.asarray(world_array) - 4e9) / 3e9 + 10

    @property
    def world_axis_object_components(self) -> Sequence[_WorldAxisComponent]:
        return (("test", 0, "value"),)

    @property
    def world_axis_object_classes(self) -> Mapping[str, _WorldAxisClass]:
        return {"test": (Quantity, (), {"unit": "Hz"})}


@pytest.fixture
def spectral_1d_ape14_wcs() -> "Spectral1DLowLevelWCS":
    return Spectral1DLowLevelWCS()


class Celestial2DLowLevelWCS(BaseLowLevelWCS):
    @property
    def pixel_n_dim(self) -> int:
        return 2

    @property
    def world_n_dim(self) -> int:
        return 2

    @property
    def world_axis_physical_types(self) -> Sequence[str | None]:
        return "pos.eq.ra", "pos.eq.dec"

    @property
    def world_axis_units(self) -> Sequence[str]:
        return "deg", "deg"

    @property
    def world_axis_names(self) -> Sequence[str]:
        return "Right Ascension", "Declination"

    @property
    def pixel_shape(self) -> tuple[int, ...] | None:
        return (6, 7)

    @property
    def pixel_bounds(self) -> Sequence[tuple[float, float] | None] | None:
        return (-1, 5), (1, 7)

    def pixel_to_world_values(self, *pixel_arrays: ArrayLike) -> Any:
        px, py = pixel_arrays
        return (-(np.asarray(px) - 5.0) * 2 + 4.0, (np.asarray(py) - 6.0) * 2)

    def world_to_pixel_values(self, *world_arrays: ArrayLike) -> Any:
        wx, wy = world_arrays
        return (-(np.asarray(wx) - 4.0) / 2 + 5.0, np.asarray(wy) / 2 + 6.0)

    @property
    def world_axis_object_components(self) -> Sequence[_WorldAxisComponent]:
        return [
            ("test", 0, "spherical.lon.degree"),
            ("test", 1, "spherical.lat.degree"),
        ]

    @property
    def world_axis_object_classes(self) -> Mapping[str, _WorldAxisClass]:
        return {"test": (SkyCoord, (), {"unit": "deg"})}


@pytest.fixture
def celestial_2d_ape14_wcs() -> "Celestial2DLowLevelWCS":
    return Celestial2DLowLevelWCS()
