# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import glob

from distutils import log
from distutils.extension import Extension

from extension_helpers import setup_helpers
from extension_helpers.utils import import_file

ERFAPKGDIR = os.path.relpath(os.path.dirname(__file__))

ERFA_SRC = os.path.abspath(os.path.join(ERFAPKGDIR, '..', '..',
                                        'cextern', 'erfa'))

SRC_FILES = glob.glob(os.path.join(ERFA_SRC, '*'))
SRC_FILES += [os.path.join(ERFAPKGDIR, filename)
              for filename in ['pav2pv.c', 'pv2pav.c', 'erfa_additions.h',
                               'ufunc.c.templ', 'core.py.templ',
                               'erfa_generator.py']]

GEN_FILES = [os.path.join(ERFAPKGDIR, 'core.py'),
             os.path.join(ERFAPKGDIR, 'ufunc.c')]


def get_extensions():

    gen = import_file(os.path.join(ERFAPKGDIR, 'erfa_generator.py'))
    gen.main(verbose=False)

    sources = [os.path.join(ERFAPKGDIR, fn)
               for fn in ("ufunc.c", "pav2pv.c", "pv2pav.c")]

    import numpy
    include_dirs = [numpy.get_include()]

    libraries = []

    if setup_helpers.use_system_library('erfa'):
        libraries.append('erfa')
    else:
        # get all of the .c files in the cextern/erfa directory
        erfafns = os.listdir(ERFA_SRC)
        sources.extend(['cextern/erfa/' + fn
                        for fn in erfafns if fn.endswith('.c')])

        include_dirs.append('cextern/erfa')

    erfa_ext = Extension(
        name="astropy._erfa.ufunc",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        language="c",)

    return [erfa_ext]


def get_external_libraries():
    return ['erfa']
