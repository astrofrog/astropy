from collections import namedtuple

import numpy as np

import astropy.units as u
from astropy.units.quantity import Quantity

StokesSymbol = namedtuple("StokesSymbol", ["symbol", "description"])


# This is table 29 in the FITS 4.0 paper
STOKES_VALUE_SYMBOL_MAP = {
    1: StokesSymbol("I", "Standard Stokes unpolarized"),
    2: StokesSymbol("Q", "Standard Stokes linear"),
    3: StokesSymbol("U", "Standard Stokes linear"),
    4: StokesSymbol("V", "Standard Stokes circular"),
    -1: StokesSymbol("RR", "Right-right circular"),
    -2: StokesSymbol("LL", "Left-left circular"),
    -3: StokesSymbol("RL", "Right-left cross-circular"),
    -4: StokesSymbol("LR", "Left-right cross-circular"),
    -5: StokesSymbol("XX", "X parallel linear"),
    -6: StokesSymbol("YY", "Y parallel linear"),
    -7: StokesSymbol("XY", "XY cross linear"),
    -8: StokesSymbol("YX", "YX cross linear"),

}


class StokesCoord(Quantity):
    """
    A representation of stokes coordinates with helpers for converting to profile names.
    """
    def __new__(cls, value, unit=None, **kwargs):
        if unit is not None and unit is not u.dimensionless_unscaled:
            raise u.UnitsError("unit should not be specified explicitly")
        return super().__new__(cls, value, unit=u.dimensionless_unscaled, **kwargs)

    @property
    def _stokes_values(self):
        """
        A representation of the coordinate as integers.
        """
        # TODO: Unbroadcast here
        return type(self)(np.round(self))

    @property
    def symbol(self):
        """
        The coordinate represented as strings
        """
        known_symbols = tuple(["?"] + [s.symbol for s in STOKES_VALUE_SYMBOL_MAP.values()])
        max_len = np.max([len(s) for s in known_symbols])

        symbolarr = np.full(self.shape, "?", dtype=f"<U{max_len}")

        for value, symbol in STOKES_VALUE_SYMBOL_MAP.items():
            symbolarr[self._stokes_values == value] = symbol.symbol

        return symbolarr

    def __repr__(self):
        arrstr = np.array2string(self.symbol.view(np.ndarray), separator=', ',
                                 prefix='  ')
        return f"<{type(self).__name__} {arrstr}>"
