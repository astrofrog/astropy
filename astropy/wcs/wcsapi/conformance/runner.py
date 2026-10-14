"""
Functions to run the checks that WCS objects conform to the shared WCS API.

The checks can be run in two ways:

* `validate_wcs` runs all the checks on a WCS object and returns the results,
  and can be used outside of a test suite.

* `parametrize_with_wcsapi_checks` is a pytest decorator which generates a
  separate test for each check and each WCS object.
"""

from functools import partial

from . import checks_v1
from .core import CheckResult, SkipCheck, default_pixel_points, low_level_wcs_for

__all__ = ["parametrize_with_wcsapi_checks", "validate_wcs", "wcsapi_checks"]

# The checks for all versions of the API. Which of these are run for a given
# version is determined by the version in which each requirement was
# introduced and, if applicable, the version in which it was removed.
ALL_CHECKS = [*checks_v1.checks]


def wcsapi_checks(version=1):
    """
    The names of the checks that apply to a given version of the WCS API.
    """
    return [check.name for check in ALL_CHECKS if check.applies_to(version)]


def _run_check(check, wcs, pixel_points=None, rtol=1e-8):
    try:
        low_level_wcs = low_level_wcs_for(wcs)
        target = wcs if check.high_level else low_level_wcs
        if check.needs_pixel_points:
            if pixel_points is None:
                pixel_points = default_pixel_points(low_level_wcs)
            if pixel_points is None:
                raise SkipCheck(
                    "the WCS does not define pixel_shape or pixel_bounds, so the "
                    "pixel positions at which to evaluate it have to be given with "
                    "pixel_points"
                )
            check.func(target, pixel_points, rtol)
        else:
            check.func(target)
    except (AssertionError, SkipCheck):
        raise
    except Exception as exc:
        # A WCS that does not conform can cause errors other than the ones we
        # check for explicitly, and these should also be reported as failures.
        raise AssertionError(
            f"unexpected {type(exc).__name__} while checking the WCS: {exc}"
        ) from exc


def validate_wcs(wcs, *, pixel_points=None, version=1, rtol=1e-8):
    """
    Check whether a WCS object conforms to the shared WCS API.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS` or `~astropy.wcs.wcsapi.BaseHighLevelWCS`
        The WCS object to check. For an object that only implements the
        high-level API, the checks of the low-level API are carried out on
        the object given by its ``low_level_wcs`` attribute.
    pixel_points : list of tuple, optional
        The pixel positions at which to evaluate the WCS, each given as a
        tuple of ``pixel_n_dim`` values. If not specified, positions are
        chosen based on ``pixel_shape`` or ``pixel_bounds``, and the checks
        that need them are skipped if neither is set.
    version : int, optional
        The version of the WCS API to check against.
    rtol : float, optional
        The relative tolerance to use when comparing coordinates.

    Returns
    -------
    results : list of `CheckResult`
        The outcome of each check, where the status is one of ``'passed'``,
        ``'failed'``, or ``'skipped'``.
    """
    results = []
    for check in ALL_CHECKS:
        if not check.applies_to(version):
            continue
        try:
            _run_check(check, wcs, pixel_points=pixel_points, rtol=rtol)
        except SkipCheck as exc:
            results.append(CheckResult(check.name, "skipped", str(exc)))
        except AssertionError as exc:
            results.append(CheckResult(check.name, "failed", str(exc)))
        else:
            results.append(CheckResult(check.name, "passed"))
    return results


def _run_check_in_pytest(wcs, *, check, pixel_points, rtol):
    import pytest

    try:
        _run_check(check, wcs, pixel_points=pixel_points, rtol=rtol)
    except SkipCheck as exc:
        pytest.skip(str(exc))


def parametrize_with_wcsapi_checks(
    wcs_objects, *, ids=None, pixel_points=None, version=1, rtol=1e-8
):
    """
    Pytest decorator which generates a test for each check and each WCS object.

    The decorated function should take the two arguments ``wcs`` and ``check``
    and call ``check(wcs)``.

    Parameters
    ----------
    wcs_objects : list of `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS objects to check.
    ids : list of str, optional
        The names to use for the WCS objects in the names of the tests.
    pixel_points : list, optional
        For each WCS object, the pixel positions at which to evaluate it, or
        `None` to choose these automatically (see `validate_wcs`).
    version : int, optional
        The version of the WCS API to check against.
    rtol : float, optional
        The relative tolerance to use when comparing coordinates.

    Examples
    --------
    ::

        @parametrize_with_wcsapi_checks([wcs1, wcs2], ids=["image", "cube"])
        def test_wcsapi_conformance(wcs, check):
            check(wcs)
    """
    import pytest

    if ids is None:
        ids = [f"{type(wcs).__name__}{i}" for i, wcs in enumerate(wcs_objects)]
    if pixel_points is None:
        pixel_points = [None] * len(wcs_objects)

    params = [
        pytest.param(
            wcs,
            partial(_run_check_in_pytest, check=check, pixel_points=points, rtol=rtol),
            id=f"{wcs_id}-{check.name}",
        )
        for wcs, wcs_id, points in zip(wcs_objects, ids, pixel_points)
        for check in ALL_CHECKS
        if check.applies_to(version)
    ]
    return pytest.mark.parametrize(("wcs", "check"), params)
