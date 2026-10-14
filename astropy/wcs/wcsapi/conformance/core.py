"""
Infrastructure for defining checks that WCS objects conform to the shared WCS
API. The checks themselves are defined in separate modules, one for each
version of the API.
"""

from dataclasses import dataclass

from astropy.wcs.wcsapi.high_level_api import BaseHighLevelWCS
from astropy.wcs.wcsapi.low_level_api import BaseLowLevelWCS

# Only the classes needed to interpret the results are public, the rest of
# this module is intended for use by the modules that define the checks.
__all__ = ["CheckResult"]


class SkipCheck(Exception):
    """
    Raised by a check that cannot be carried out for a given WCS object.
    """


@dataclass(frozen=True)
class Check:
    """
    A single requirement of the WCS API along with the function that tests it.
    """

    name: str
    func: callable
    # The version of the WCS API in which the requirement was introduced
    version: int
    # The version of the WCS API from which the requirement no longer applies,
    # or None if it applies to all versions from the one it was introduced in.
    removed_in: int | None
    # Whether the check needs pixel positions at which to evaluate the WCS
    needs_pixel_points: bool
    # Whether the check is for the low-level API or the high-level API
    high_level: bool

    def applies_to(self, version):
        """
        Whether the requirement is part of a given version of the WCS API.
        """
        return self.version <= version and (
            self.removed_in is None or version < self.removed_in
        )


class CheckCollection(list):
    """
    The checks for the requirements introduced in a given version of the API.

    Parameters
    ----------
    version : int
        The version of the WCS API in which the requirements were introduced.
    """

    def __init__(self, version):
        super().__init__()
        self.version = version

    def register(self, *, needs_pixel_points=False, high_level=False, removed_in=None):
        """
        Decorator to add a function to the collection.

        The function should take the WCS object as its only argument, or if
        ``needs_pixel_points`` is set, the WCS object, a list of pixel positions,
        and a relative tolerance. It should call `require` for each condition
        that has to be satisfied.

        Checks are for the low-level API unless ``high_level`` is set. Checks
        of the low-level API are passed the low-level object, which for an
        object that only implements the high-level API is the one given by its
        ``low_level_wcs`` attribute.

        If a later version of the API drops or changes a requirement,
        ``removed_in`` should be set to that version, so that the check is no
        longer run for objects that implement that version or a later one. A
        requirement that changes is then defined as a new check in the
        collection for the later version.
        """

        def decorator(func):
            name = func.__name__.removeprefix("check_")
            self.append(
                Check(
                    name,
                    func,
                    self.version,
                    removed_in,
                    needs_pixel_points,
                    high_level,
                )
            )
            return func

        return decorator


@dataclass(frozen=True)
class CheckResult:
    """
    The outcome of running a single check on a WCS object.
    """

    name: str
    status: str
    message: str = ""


def require(condition, message):
    """
    Raise an `AssertionError` with the given message if a condition is not met.

    This is used rather than ``assert`` so that the checks are still carried
    out when Python is run with optimizations enabled.
    """
    if not condition:
        raise AssertionError(message)


def low_level_wcs_for(wcs):
    """
    The object exposing the low-level API for a given WCS object.

    This is the object itself unless it only implements the high-level API,
    in which case it is the object given by its ``low_level_wcs`` attribute.
    """
    if isinstance(wcs, BaseHighLevelWCS) and not isinstance(wcs, BaseLowLevelWCS):
        return wcs.low_level_wcs
    return wcs


def as_tuple(values, n_dim):
    # Methods of the low-level API return a single scalar or array instead of
    # a tuple when there is only one output dimension.
    if n_dim == 1 and not isinstance(values, (tuple, list)):
        return (values,)
    return tuple(values)


def default_pixel_points(wcs):
    """
    Pixel positions at which to evaluate a WCS, based on its shape or bounds.
    """
    if wcs.pixel_shape is not None:
        ranges = [(0, n - 1) for n in wcs.pixel_shape]
    elif wcs.pixel_bounds is not None and None not in wcs.pixel_bounds:
        ranges = list(wcs.pixel_bounds)
    else:
        return None
    # Two positions that are away from the edges, that do not lie on the same
    # row or column, and that are not half-way between two pixel centers
    # since the way in which these are rounded is not well defined.
    return [
        tuple(float(lo + frac * (hi - lo)) for lo, hi in ranges)
        for frac in (0.2718, 0.6180)
    ]
