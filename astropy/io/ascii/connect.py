# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This file connects the readers/writers to the astropy.table.Table class

from __future__ import absolute_import, division, print_function

import re

from ..registry import BaseIO
from ...table import Table
from ...extern.six.moves import zip

__all__ = []


# Generic
# =======


def read_asciitable(filename, **kwargs):
    from .ui import read
    return read(filename, **kwargs)


def write_asciitable(table, filename, **kwargs):
    from .ui import write
    return write(table, filename, **kwargs)


class ASCIITableIO(BaseIO):

    _format_name = 'ascii'
    _supported_class = Table

    @staticmethod
    def read(format, filename, **kwargs):
        from .ui import read
        format = re.sub(r'^ascii\.', '', format)
        return read(filename, format=format, **kwargs)

    @staticmethod
    def write(format, table, filename, **kwargs):
        from .ui import write
        format = re.sub(r'^ascii\.', '', format)
        return write(table, filename, format=format, **kwargs)

    @staticmethod
    def identify(suffix, origin, filepath, fileobj, *args, **kwargs):
        return filepath is not None and filepath.endswith(suffix)
