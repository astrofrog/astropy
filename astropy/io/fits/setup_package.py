# Licensed under a 3-clause BSD style license - see PYFITS.rst

import os
from collections import defaultdict

from distutils.core import Extension
from glob import glob

import numpy

from extension_helpers import setup_helpers


def _get_compression_extension():

    cfg = defaultdict(list)
    cfg['include_dirs'].append(numpy.get_include())
    cfg['sources'].append(os.path.join(os.path.dirname(__file__), 'src', 'compressionmodule.c'))

    if int(os.environ.get('ASTROPY_USE_SYSTEM_CFITSIO', 0)) or int(os.environ.get('ASTROPY_USE_SYSTEM_ALL', 0)):
        cfg.update(setup_helpers.pkg_config(['cfitsio'], ['cfitsio']))
    else:
        if setup_helpers.get_compiler_option() == 'msvc':
            # These come from the CFITSIO vcc makefile, except the last
            # which ensures on windows we do not include unistd.h (in regular
            # compilation of cfitsio, an empty file would be generated)
            cfg['extra_compile_args'].extend(
                ['/D', '"WIN32"',
                 '/D', '"_WINDOWS"',
                 '/D', '"_MBCS"',
                 '/D', '"_USRDLL"',
                 '/D', '"_CRT_SECURE_NO_DEPRECATE"',
                 '/D', '"FF_NO_UNISTD_H"'])
        else:
            cfg['extra_compile_args'].extend([
                '-Wno-declaration-after-statement'
            ])

            if not int(os.environ.get('ASTROPY_DEBUG', 0)):
                # these switches are to silence warnings from compiling CFITSIO
                # For full silencing, some are added that only are used in
                # later versions of gcc (versions approximate; see #6474)
                cfg['extra_compile_args'].extend([
                    '-Wno-strict-prototypes',
                    '-Wno-unused',
                    '-Wno-uninitialized',
                    '-Wno-unused-result',  # gcc >~4.8
                    '-Wno-misleading-indentation',  # gcc >~7.2
                    '-Wno-format-overflow',  # gcc >~7.2
                ])

        cfitsio_lib_path = os.path.join('cextern', 'cfitsio', 'lib')
        cfitsio_zlib_path = os.path.join('cextern', 'cfitsio', 'zlib')
        cfitsio_files = glob(os.path.join(cfitsio_lib_path, '*.c'))
        cfitsio_zlib_files = glob(os.path.join(cfitsio_zlib_path, '*.c'))
        cfg['include_dirs'].append(cfitsio_lib_path)
        cfg['include_dirs'].append(cfitsio_zlib_path)
        cfg['sources'].extend(cfitsio_files)
        cfg['sources'].extend(cfitsio_zlib_files)

    return Extension('astropy.io.fits.compression', **cfg)


def get_extensions():
    return [_get_compression_extension()]
