# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from distutils.extension import Extension

C_CONVOLVE_PKGDIR = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    lib_convolve_none = Extension(name='astropy.convolution._convolve_boundary_none',
                                  sources=[os.path.join(C_CONVOLVE_PKGDIR, 'src/convolve_boundary_none.c')],
                                  include_dirs=["numpy"],
                                  language='c')

    lib_convolve_padded = Extension(name='astropy.convolution._convolve_boundary_padded',
                                    sources=[os.path.join(C_CONVOLVE_PKGDIR, 'src/convolve_boundary_padded.c')],
                                    include_dirs=["numpy"],
                                    language='c')

    return [lib_convolve_none, lib_convolve_padded]
