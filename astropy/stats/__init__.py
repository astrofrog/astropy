# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This subpackage contains statistical tools provided for or used by Astropy.

While the `scipy.stats` package contains a wide range of statistical
tools, it is a general-purpose package, and is missing some that are
particularly useful to astronomy or are used in an atypical way in
astronomy. This package is intended to provide such functionality, but
*not* to replace `scipy.stats` if its implementation satisfies
astronomers' needs.

"""

from .bayesian_blocks import *
from .biweight import *
from .bls import *
from .circstats import *
from .funcs import *
from .histogram import *
from .info_theory import *
from .jackknife import *
from .lombscargle import *
from .sigma_clipping import *
from .spatial import *
