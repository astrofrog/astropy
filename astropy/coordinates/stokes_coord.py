from collections import namedtuple
from contextlib import contextmanager
from copy import copy
from typing import Dict, List

import numpy as np

import astropy.units as u
from astropy.units.quantity import Quantity

StokesSymbol = namedtuple("StokesSymbol", ["symbol", "description"], defaults=[""])

# This is table 29 in the FITS 4.0 paper
FITS_STOKES_VALUE_SYMBOL_MAP = {
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

STOKES_VALUE_SYMBOL_MAP = copy(FITS_STOKES_VALUE_SYMBOL_MAP)


@contextmanager
def custom_stokes_symbol_mapping(mapping: Dict[int, StokesSymbol], replace: bool = False):
    """
    Add a custom set of mappings from values to Stokes symbols.

    Parameters
    ----------
    mappings
        A list of dictionaries with custom mappings between values (integers)
        and `StokesSymbol` classes.
    replace
        Replace all mappings with this one.
    """
    global STOKES_VALUE_SYMBOL_MAP

    original_mapping = copy(STOKES_VALUE_SYMBOL_MAP)
    if not replace:
        STOKES_VALUE_SYMBOL_MAP = {**original_mapping, **mapping}
    else:
        STOKES_VALUE_SYMBOL_MAP = mapping
    yield
    STOKES_VALUE_SYMBOL_MAP = original_mapping


class StokesCoord(Quantity):
    """
    A representation of stokes coordinates with helpers for converting to profile names.
    """
    def __new__(cls, value, unit=None, **kwargs):
        if unit is not None and unit is not u.dimensionless_unscaled:
            raise u.UnitsError("unit should not be specified explicitly")

        value_as_array = np.array(value, copy=False, subok=True)
        if value_as_array.dtype.kind == "U":
            return cls.from_symbols(value_as_array)

        return super().__new__(cls, value, unit=u.dimensionless_unscaled, **kwargs)

    @classmethod
    def from_symbols(cls, value):
        """
        Construct a StokesCoord from strings representing the stokes symbols.
        """
        values_array = np.full_like(value, np.nan, dtype=float, subok=False)
        for stokes_value, symbol in STOKES_VALUE_SYMBOL_MAP.items():
            values_array[value == symbol.symbol] = stokes_value

        if (nan_values := np.isnan(values_array)).any():
            raise ValueError(
                f"Unknown stokes symbols present in the input array: {np.unique(value[nan_values])}"
            )

        return super().__new__(cls, values_array, unit=u.dimensionless_unscaled)

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

    def __str__(self):
        arrstr = np.array2string(self.symbol.view(np.ndarray), separator=', ',
                                 prefix='  ')
        return f"{type(self).__name__}({arrstr})"

    def __repr__(self):
        arrstr = np.array2string(self.symbol.view(np.ndarray), separator=', ',
                                 prefix='  ')
        return f"<{type(self).__name__} {arrstr}>"
