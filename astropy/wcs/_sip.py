# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pure-Python ``Sip`` class.

The SIP distortion polynomials are evaluated by WCSLIB.  This class holds the
coefficient arrays, translates them into the WCSLIB distortion keyrecords
(``NAXES``/``OFFSET``/``SIP.FWD``), and delegates the actual transform to the
generic compiled engine ``_wcs.Distortion`` (a thin ``disprm`` binding that
carries no distortion-specific maths).  The public Python API is unchanged.
"""

import numpy as np

from . import _wcs

__all__ = ["Sip"]


def _as_coeffs(arr):
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=np.double)


def _sip_disprm(cx, cy, crpix):
    """Build a 2-axis SIP ``_wcs.Distortion`` from two coefficient matrices.

    ``cx``/``cy`` multiply axes 1/2.  This is the entire SIP-specific
    "knowledge" -- which terms exist and that the renormalisation offset is
    CRPIX -- expressed as plain WCSLIB DPja keyrecords; WCSLIB turns them into
    a TPD and evaluates it.
    """
    records = []
    for axis, c in ((1, cx), (2, cy)):
        records.append(("NAXES", axis, 2.0))
        records.append(("OFFSET.1", axis, float(crpix[0])))
        records.append(("OFFSET.2", axis, float(crpix[1])))
        order = c.shape[0] - 1
        for p in range(order + 1):
            for q in range(order + 1):
                v = c[p, q]
                if v != 0.0:
                    records.append((f"SIP.FWD.{p}_{q}", axis, float(v)))
    return _wcs.Distortion(2, ["SIP", "SIP"], records)


class Sip:
    """
    The `Sip` class provides access to the Simple Imaging Polynomial (SIP)
    distortion (https://fits.gsfc.nasa.gov/registry/sip.html).

    Parameters
    ----------
    a, b, ap, bp : ndarray or None
        The ``A``/``B`` (forward) and ``AP``/``BP`` (reverse) coefficient
        matrices, each ``(order + 1, order + 1)``.  ``ap``/``bp`` may be `None`.
    crpix : array-like
        The reference pixel ``(CRPIX1, CRPIX2)``.
    """

    def __init__(self, a, b, ap, bp, crpix):
        self._a = _as_coeffs(a)
        self._b = _as_coeffs(b)
        self._ap = _as_coeffs(ap)
        self._bp = _as_coeffs(bp)
        self._crpix = np.ascontiguousarray(crpix, dtype=np.double)

        # Compiled transform engines (generic disprm bindings).  The reverse
        # SIP polynomial (AP/BP) is evaluated as a *forward* TPD, matching the
        # historical direct (non-iterative) AP/BP evaluation.
        self._fwd = (
            None if self._a is None else _sip_disprm(self._a, self._b, self._crpix)
        )
        self._rev = (
            None if self._ap is None else _sip_disprm(self._ap, self._bp, self._crpix)
        )

    # -- coefficient data ---------------------------------------------------
    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def ap(self):
        return self._ap

    @property
    def bp(self):
        return self._bp

    @property
    def crpix(self):
        return self._crpix

    @property
    def a_order(self):
        return 0 if self._a is None else self._a.shape[0] - 1

    @property
    def b_order(self):
        return 0 if self._b is None else self._b.shape[0] - 1

    @property
    def ap_order(self):
        return 0 if self._ap is None else self._ap.shape[0] - 1

    @property
    def bp_order(self):
        return 0 if self._bp is None else self._bp.shape[0] - 1

    # -- transforms ---------------------------------------------------------
    # disp2x() returns the full corrected coordinate (raw + polynomial); the
    # CRPIX subtraction and the FITS origin offset are applied here, exactly
    # reproducing the historical C ``Sip.pix2foc``/``foc2pix`` behaviour.
    def pix2foc(self, pixcrd, origin):
        if self._fwd is None:
            raise ValueError(
                "SIP object does not have coefficients for pix2foc "
                "transformation (A and B)"
            )
        pix = np.asarray(pixcrd, dtype=np.double)
        off = 1 - origin
        foc = self._fwd.transform(pix + off)
        return foc - self._crpix - off

    def foc2pix(self, foccrd, origin):
        if self._rev is None:
            raise ValueError(
                "SIP object does not have coefficients for foc2pix "
                "transformation (AP and BP)"
            )
        foc = np.asarray(foccrd, dtype=np.double)
        off = 1 - origin
        return self._rev.transform(foc + off + self._crpix) - off

    # -- copy / pickle ------------------------------------------------------
    def __reduce__(self):
        return (self.__class__, (self._a, self._b, self._ap, self._bp, self._crpix))

    def __copy__(self):
        return self.__class__(self._a, self._b, self._ap, self._bp, self._crpix)

    def __deepcopy__(self, memo):
        from copy import deepcopy

        return self.__class__(
            deepcopy(self._a, memo),
            deepcopy(self._b, memo),
            deepcopy(self._ap, memo),
            deepcopy(self._bp, memo),
            deepcopy(self._crpix, memo),
        )
