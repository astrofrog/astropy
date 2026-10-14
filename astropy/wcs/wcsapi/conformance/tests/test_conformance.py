import numpy as np
import pytest

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSMixin, HighLevelWCSWrapper, SlicedLowLevelWCS
from astropy.wcs.wcsapi.conformance import (
    parametrize_with_wcsapi_checks,
    validate_wcs,
    wcsapi_checks,
)
from astropy.wcs.wcsapi.conformance.core import CheckCollection
from astropy.wcs.wcsapi.conftest import Celestial2DLowLevelWCS, Spectral1DLowLevelWCS


def celestial_2d_fitswcs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.cdelt = -0.01, 0.01
    wcs.wcs.crval = 10.0, 20.0
    wcs.wcs.crpix = 30.0, 40.0
    wcs.pixel_shape = (60, 80)
    return wcs


def spectral_cube_3d_fitswcs():
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN", "FREQ"
    wcs.wcs.cunit = "deg", "deg", "Hz"
    wcs.wcs.cdelt = -0.01, 0.01, 3.0e9
    wcs.wcs.crval = 10.0, 20.0, 4.0e9
    wcs.wcs.crpix = 30.0, 40.0, 11.0
    wcs.pixel_shape = (60, 80, 20)
    return wcs


def time_cube_4d_fitswcs():
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN", "FREQ", "TIME"
    wcs.wcs.cunit = "deg", "deg", "Hz", "s"
    wcs.wcs.cdelt = -0.01, 0.01, 3.0e9, 1
    wcs.wcs.crval = 10.0, 20.0, 4.0e9, 3
    wcs.wcs.crpix = 30.0, 40.0, 11.0, 11.0
    wcs.wcs.mjdref = (55000, 0)
    wcs.pixel_shape = (60, 80, 20, 10)
    return wcs


def sliced_wrapper():
    return SlicedLowLevelWCS(
        spectral_cube_3d_fitswcs(), (slice(None), 3, slice(10, 40))
    )


@parametrize_with_wcsapi_checks(
    [
        celestial_2d_fitswcs(),
        spectral_cube_3d_fitswcs(),
        time_cube_4d_fitswcs(),
        spectral_cube_3d_fitswcs()[5],
        sliced_wrapper(),
        HighLevelWCSWrapper(sliced_wrapper()),
        Celestial2DLowLevelWCS(),
        Spectral1DLowLevelWCS(),
    ],
    ids=[
        "fits_celestial",
        "fits_cube",
        "fits_time_cube",
        "fits_cube_sliced",
        "sliced_wrapper",
        "high_level_wrapper",
        "custom_celestial",
        "custom_spectral",
    ],
    pixel_points=[None, None, None, None, None, None, None, [(3.2,), (12.7,)]],
)
def test_wcsapi_conformance(wcs, check):
    check(wcs)


def test_validate_wcs_passes():
    results = validate_wcs(spectral_cube_3d_fitswcs())
    assert [result.name for result in results] == wcsapi_checks()
    assert {result.status for result in results} == {"passed"}


def test_validate_wcs_without_pixel_points():
    # This WCS does not define a shape or bounds, so the checks that need to
    # evaluate the WCS are skipped unless positions are given explicitly.
    wcs = Spectral1DLowLevelWCS()
    statuses = {result.name: result.status for result in validate_wcs(wcs)}
    assert statuses["dimensions"] == "passed"
    assert statuses["world_to_pixel_values"] == "skipped"

    results = validate_wcs(wcs, pixel_points=[(3.2,), (12.7,)])
    assert {result.status for result in results} == {"passed"}


# Each of the following classes breaks the API in a single way


class NotRegistered:
    # An object which provides the whole API, but which neither inherits from
    # the low-level base class nor is registered with it.

    def __init__(self):
        self._wcs = Celestial2DLowLevelWCS()

    def __getattr__(self, name):
        return getattr(self._wcs, name)


class MissingMethod(NotRegistered):
    def __getattr__(self, name):
        if name == "world_to_array_index_values":
            raise AttributeError(name)
        return super().__getattr__(name)


class NonIntegerDimensions(Celestial2DLowLevelWCS):
    @property
    def pixel_n_dim(self):
        return 2.0


class InconsistentShapes(Celestial2DLowLevelWCS):
    @property
    def array_shape(self):
        return (6, 7)


class NonIntegerShape(Celestial2DLowLevelWCS):
    @property
    def pixel_shape(self):
        return (6.0, 7.0)


class InvertedBounds(Celestial2DLowLevelWCS):
    @property
    def pixel_bounds(self):
        return (5, -1), (1, 7)


class InvalidPhysicalType(Celestial2DLowLevelWCS):
    @property
    def world_axis_physical_types(self):
        return "pos.eq.ra", "declination"


class WrongNumberOfUnits(Celestial2DLowLevelWCS):
    @property
    def world_axis_units(self):
        return ("deg",)


class InvalidUnit(Celestial2DLowLevelWCS):
    @property
    def world_axis_units(self):
        return "deg", "degrees_north"


class WrongNumberOfNames(Celestial2DLowLevelWCS):
    @property
    def pixel_axis_names(self):
        return ("x",)


class NonBooleanCorrelationMatrix(Celestial2DLowLevelWCS):
    @property
    def axis_correlation_matrix(self):
        return np.ones((2, 2), dtype=int)


class UnderstatedCorrelationMatrix(Celestial2DLowLevelWCS):
    @property
    def axis_correlation_matrix(self):
        return np.array([[True, False], [False, False]])


class NonBooleanSerializedClasses(Celestial2DLowLevelWCS):
    @property
    def serialized_classes(self):
        return 0


class UnknownObjectName(Celestial2DLowLevelWCS):
    @property
    def world_axis_object_components(self):
        return [
            ("celestial", 0, "spherical.lon.degree"),
            ("celestial", 1, "spherical.lat.degree"),
        ]


class ClassNotSerialized(Celestial2DLowLevelWCS):
    @property
    def serialized_classes(self):
        return True


class SwappedObjectComponents(Celestial2DLowLevelWCS):
    @property
    def world_axis_object_components(self):
        return [
            ("test", 0, "spherical.lat.degree"),
            ("test", 1, "spherical.lon.degree"),
        ]


class Unconvertible:
    # A class for world objects that cannot be initialized from an instance

    def __init__(self, lon, lat, unit=None):
        self.spherical = SkyCoord(float(lon), float(lat), unit=unit).spherical


class UnconvertibleObjectClass(Celestial2DLowLevelWCS):
    @property
    def world_axis_object_classes(self):
        return {"test": (Unconvertible, (), {"unit": "deg"})}


class ArrayOutputForScalars(Celestial2DLowLevelWCS):
    def pixel_to_world_values(self, px, py):
        wx, wy = super().pixel_to_world_values(px, py)
        return np.atleast_1d(wx), np.atleast_1d(wy)


class InconsistentInverse(Celestial2DLowLevelWCS):
    def world_to_pixel_values(self, wx, wy):
        px, py = super().world_to_pixel_values(wx, wy)
        return px + 1, py


class FlattenedArrays(Celestial2DLowLevelWCS):
    def pixel_to_world_values(self, px, py):
        wx, wy = super().pixel_to_world_values(px, py)
        return (wx.ravel(), wy.ravel()) if np.ndim(wx) > 1 else (wx, wy)


class ArrayIndicesInPixelOrder(Celestial2DLowLevelWCS):
    def array_index_to_world_values(self, *index_arrays):
        return self.pixel_to_world_values(*index_arrays)


class UnroundedArrayIndices(Celestial2DLowLevelWCS):
    def world_to_array_index_values(self, wx, wy):
        return self.world_to_pixel_values(wx, wy)[::-1]


class TruncatedArrayIndices(Celestial2DLowLevelWCS):
    def world_to_array_index_values(self, wx, wy):
        px, py = self.world_to_pixel_values(wx, wy)
        return np.asarray(py, dtype=int), np.asarray(px, dtype=int)


class NoLowLevelWCS(Celestial2DLowLevelWCS, HighLevelWCSMixin):
    # This is checked as is rather than through its low_level_wcs attribute,
    # since it also inherits from the low-level base class.
    @property
    def low_level_wcs(self):
        return None


class HighLevelArrayIndicesInPixelOrder(Celestial2DLowLevelWCS, HighLevelWCSMixin):
    @property
    def low_level_wcs(self):
        return self

    def array_index_to_world(self, *index_arrays):
        return self.pixel_to_world(*index_arrays)


@pytest.mark.parametrize(
    ("wcs_class", "failed", "others_fail"),
    [
        (NotRegistered, {"low_level_base_class"}, True),
        (MissingMethod, {"low_level_attributes"}, True),
        (NonIntegerDimensions, {"dimensions"}, True),
        (InconsistentShapes, {"array_shape"}, False),
        (NonIntegerShape, {"array_shape"}, False),
        (InvertedBounds, {"pixel_bounds"}, False),
        (InvalidPhysicalType, {"world_axis_physical_types"}, False),
        (WrongNumberOfUnits, {"world_axis_units"}, False),
        (InvalidUnit, {"world_axis_units"}, False),
        (WrongNumberOfNames, {"axis_names"}, False),
        (NonBooleanCorrelationMatrix, {"axis_correlation_matrix"}, False),
        (UnderstatedCorrelationMatrix, {"axis_correlation_matrix_dependencies"}, False),
        (NonBooleanSerializedClasses, {"serialized_classes"}, False),
        (UnknownObjectName, {"world_axis_object_components"}, True),
        (ClassNotSerialized, {"world_axis_object_classes"}, True),
        (SwappedObjectComponents, {"pixel_to_world", "world_to_pixel"}, True),
        (UnconvertibleObjectClass, {"world_object_conversion"}, True),
        (ArrayOutputForScalars, {"pixel_to_world_values"}, True),
        (InconsistentInverse, {"world_to_pixel_values", "world_to_pixel"}, True),
        (FlattenedArrays, {"array_input"}, False),
        (ArrayIndicesInPixelOrder, {"array_index_to_world_values"}, False),
        (UnroundedArrayIndices, {"world_to_array_index_values"}, False),
        (TruncatedArrayIndices, {"world_to_array_index_values"}, False),
        (NoLowLevelWCS, {"high_level_attributes"}, True),
        (HighLevelArrayIndicesInPixelOrder, {"array_index_to_world"}, False),
    ],
)
def test_validate_wcs_failures(wcs_class, failed, others_fail):
    # We check that each problem is detected by the checks we expect. Some
    # problems unavoidably cause other checks to fail too, for example the
    # high-level API cannot work if the definition of the high-level objects
    # is not valid, and for the other problems we check that no other checks
    # fail.
    results = validate_wcs(wcs_class())
    actual = {result.name for result in results if result.status == "failed"}
    if others_fail:
        assert failed < actual
    else:
        assert failed == actual
    for result in results:
        if result.status == "failed":
            assert result.message


def test_validate_wcs_high_level_only():
    # For an object that only implements the high-level API, the checks of the
    # low-level API are carried out on its low_level_wcs attribute.
    wcs = HighLevelWCSWrapper(UnderstatedCorrelationMatrix())
    failed = [result.name for result in validate_wcs(wcs) if result.status == "failed"]
    assert failed == ["axis_correlation_matrix_dependencies"]


def test_check_versions():
    # Requirements can be introduced in one version of the API and removed in
    # a later one, in which case the check should only apply to the versions
    # in between.
    checks_v1 = CheckCollection(version=1)
    checks_v2 = CheckCollection(version=2)

    @checks_v1.register()
    def check_kept(wcs):
        pass

    @checks_v1.register(removed_in=3)
    def check_removed(wcs):
        pass

    @checks_v2.register(needs_pixel_points=True)
    def check_added(wcs, pixel_points, rtol):
        pass

    def names(version):
        return [c.name for c in [*checks_v1, *checks_v2] if c.applies_to(version)]

    assert names(1) == ["kept", "removed"]
    assert names(2) == ["kept", "removed", "added"]
    assert names(3) == ["kept", "added"]
