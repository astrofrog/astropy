import numpy as np
import pytest
from numpy.testing import assert_equal

from astropy import units as u
from astropy.coordinates.stokes_coord import StokesCoord, StokesSymbol, custom_stokes_symbol_mapping


def test_scalar():
    sk = StokesCoord(1)
    assert repr(sk) == '<StokesCoord Q>'
    assert sk.value == 1.
    assert sk.symbol == 'Q'


def test_vector():
    # This also checks that floats are rounded when converting
    # to strings
    values = [0.2, 0.8, 1., 1.2, 1.8, 2.4]
    sk = StokesCoord(values)
    assert repr(sk) == '<StokesCoord [I, Q, Q, U, U]>'
    assert_equal(sk.value, values)
    assert_equal(sk.symbol, np.array(['I', 'Q', 'Q', 'U', 'U']))


def test_unit():
    StokesCoord(1, unit=u.one)
    for unit in [u.radian, u.deg, u.Hz]:
        with pytest.raises(u.UnitsError, match='unit should not be specified explicitly'):
            StokesCoord(1, unit=unit)


def test_undefined():
    sk = StokesCoord(np.arange(-10, 7))
    assert_equal(sk.symbol,
                 np.array(['?', '?',
                           'YX', 'XY', 'YY', 'XX',
                           'LR', 'RL', 'LL', 'RR',
                           '?', 'I', 'Q', 'U', 'V',
                           '?', '?']))


def test_custom_symbol_mapping():

    custom_mapping = {
        10000: StokesSymbol('A'),
        10001: StokesSymbol('B'),
        10002: StokesSymbol('C'),
        10003: StokesSymbol('D'),
    }

    # Check that we can supply a custom mapping
    with custom_stokes_symbol_mapping(custom_mapping):
        values = [0.2, 1.2, 10000.1, 10002.4]
        sk1 = StokesCoord(values)
        assert repr(sk1) == '<StokesCoord [I, Q, A, C]>'
        assert_equal(sk1.value, values)
        assert_equal(sk1.symbol, np.array(['I', 'Q', 'A', 'C']))

    # Check that the mapping is still active outside the context manager
    assert_equal(sk1.symbol, np.array(['I', 'Q', 'A', 'C']))

    # But not for new StokesCoords
    sk2 = StokesCoord(values)
    assert_equal(sk2.symbol, np.array(['I', 'Q', '?', '?']))


def test_custom_symbol_mapping_overlap():

    # Make a custom mapping that overlaps with some of the existing values

    custom_mapping = {
        2: StokesSymbol('A'),
        3: StokesSymbol('B'),
        4: StokesSymbol('C'),
        5: StokesSymbol('D'),
    }

    with custom_stokes_symbol_mapping(custom_mapping):
        sk = StokesCoord(np.arange(6))
        assert_equal(sk.symbol, np.array(['I', 'Q', 'A', 'B', 'C', 'D']))


def test_custom_symbol_mapping_replace():

    # Check that we can replace the mapping completely

    custom_mapping = {
        2: StokesSymbol('A'),
        3: StokesSymbol('B'),
        4: StokesSymbol('C'),
        5: StokesSymbol('D'),
    }

    with custom_stokes_symbol_mapping(custom_mapping, replace=True):
        sk = StokesCoord(np.arange(6))
        assert_equal(sk.symbol, np.array(['?', '?', 'A', 'B', 'C', 'D']))
