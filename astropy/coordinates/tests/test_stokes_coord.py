import numpy as np
import pytest
from numpy.testing import assert_equal

from astropy import units as u
from astropy.coordinates.stokes_coord import StokesCoord


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
