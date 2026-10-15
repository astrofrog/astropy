# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Unit tests for the zenithal plane-to-plane accelerator (``astropy.wcs._accel``)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy.wcs import WCS
from astropy.wcs._accel import (
    SUPPORTED_PROJECTIONS,
    apply_transform,
    compute_transform,
)

ZENITHAL = sorted(SUPPORTED_PROJECTIONS)


def make_wcs(proj, crval, crpix, cdelt, rot=0.0, frame="RA"):
    w = WCS(naxis=2)
    if frame == "RA":
        w.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
    else:
        w.wcs.ctype = [f"GLON-{proj}", f"GLAT-{proj}"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    w.wcs.cd = np.array([[cdelt * c, -cdelt * s], [cdelt * s, cdelt * c]])
    w.wcs.set()
    return w


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("proj_in", ZENITHAL)
@pytest.mark.parametrize("proj_out", ZENITHAL)
def test_matches_full_pipeline(proj_in, proj_out):
    # The validation recipe: agree with the full WCS pipeline for every pair.
    w1 = make_wcs(proj_in, [266.4, -29.0], [256, 256], -0.0015)
    w2 = make_wcs(proj_out, [266.7, -28.7], [300, 280], -0.0011, rot=17)
    gx, gy = np.meshgrid(np.linspace(-700, 700, 25), np.linspace(-700, 700, 25))
    px = (256 + gx).ravel()
    py = (256 + gy).ravel()

    fx, fy = apply_transform(compute_transform(w1, w2), px, py)
    sky = w1.pixel_to_world(px, py)
    rx, ry = w2.world_to_pixel(sky)
    assert_allclose(fx, rx, atol=1e-8)
    assert_allclose(fy, ry, atol=1e-8)


def test_tan_tan_uses_single_matrix():
    w1 = make_wcs("TAN", [266.4, -29.0], [256, 256], -0.0015)
    w2 = make_wcs("TAN", [266.7, -28.7], [300, 280], -0.0011, rot=17)
    assert compute_transform(w1, w2).matrix is not None  # collapses to a homography
    # any non-TAN leg keeps the general (rescale) path
    w3 = make_wcs("ZEA", [266.7, -28.7], [300, 280], -0.0011)
    assert compute_transform(w1, w3).matrix is None


def test_inverse_round_trip():
    w1 = make_wcs("ZEA", [266.4, -29.0], [256, 256], -0.0015)
    w2 = make_wcs("STG", [266.7, -28.7], [300, 280], -0.0011, rot=17)
    px = np.array([100.0, 256.0, 400.0, 50.0])
    py = np.array([120.0, 256.0, 380.0, 470.0])
    fx, fy = apply_transform(compute_transform(w1, w2), px, py)
    bx, by = apply_transform(compute_transform(w2, w1), fx, fy)
    assert_allclose(bx, px, atol=1e-9)
    assert_allclose(by, py, atol=1e-9)


@pytest.mark.filterwarnings("ignore")
def test_out_of_domain_is_masked():
    # SIN can only image rho < 90 deg; pixels beyond that have no preimage and
    # must come back NaN, matching the full pipeline's validity.
    w1 = make_wcs("SIN", [266.4, -29.0], [256, 256], -0.1)  # wide, 0.1 deg/pix
    w2 = make_wcs("TAN", [266.4, -29.0], [256, 256], -0.1)
    px = np.array([256.0, 256.0 + 700])  # ~70 deg out -> psi > 1, invalid
    py = np.array([256.0, 256.0])
    fx, _ = apply_transform(compute_transform(w1, w2), px, py)
    assert np.isfinite(fx[0])
    assert np.isnan(fx[1])

    # validity mask agrees with the full pipeline across a wide grid
    gx, gy = np.meshgrid(np.linspace(-900, 900, 40), np.linspace(-900, 900, 40))
    px = (256 + gx).ravel()
    py = (256 + gy).ravel()
    fx, fy = apply_transform(compute_transform(w1, w2), px, py)
    rx, ry = w2.world_to_pixel(w1.pixel_to_world(px, py))
    fast_ok = np.isfinite(fx) & np.isfinite(fy)
    ref_ok = np.isfinite(rx) & np.isfinite(ry)
    assert np.array_equal(fast_ok, ref_ok)
    assert_allclose(fx[fast_ok], rx[fast_ok], atol=1e-7)


def test_beyond_horizon_is_masked():
    # A point in the hemisphere the output plane cannot image (w <= 0) is NaN.
    w1 = make_wcs("TAN", [10.0, 0.0], [256, 256], -0.001)
    w2 = make_wcs("TAN", [120.0, 0.0], [256, 256], -0.001)  # ~110 deg away
    fx, fy = apply_transform(compute_transform(w1, w2), 256.0, 256.0)
    assert np.isnan(fx) and np.isnan(fy)


def test_rejects_unsupported_projection():
    w1 = make_wcs("TAN", [266.4, -29.0], [256, 256], -0.0015)
    w2 = make_wcs("CAR", [266.7, -28.7], [300, 280], -0.0011)
    with pytest.raises(ValueError, match="zenithal"):
        compute_transform(w1, w2)


def test_rejects_frame_mismatch():
    w1 = make_wcs("TAN", [266.4, -29.0], [256, 256], -0.0015, frame="RA")
    w2 = make_wcs("TAN", [120.0, 30.0], [300, 280], -0.0011, frame="GLON")
    with pytest.raises(ValueError, match="same celestial frame"):
        compute_transform(w1, w2)


def test_apply_transform_namespace_agnostic():
    jnp = pytest.importorskip("jax.numpy")
    w1 = make_wcs("TAN", [266.4, -29.0], [256, 256], -0.0015)
    w2 = make_wcs("ZEA", [266.7, -28.7], [300, 280], -0.0011, rot=17)
    px = np.linspace(80, 440, 50)
    py = np.linspace(70, 450, 50)
    transform = compute_transform(w1, w2)

    nx, ny = apply_transform(transform, px, py)
    jx, jy = apply_transform(transform, jnp.asarray(px), jnp.asarray(py), xp=jnp)
    assert jx.__array_namespace__().__name__ == "jax.numpy"
    assert_allclose(np.asarray(jx), nx, atol=1e-3)  # jax defaults to float32
    assert_allclose(np.asarray(jy), ny, atol=1e-3)
