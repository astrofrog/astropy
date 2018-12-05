# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" An extensible ASCII table reader and writer.

"""


from .basic import (Basic, BasicHeader, BasicData,
                    Rdb,
                    Csv,
                    Tab,
                    NoHeader,
                    CommentedHeader)
from .cds import Cds
from .core import (InconsistentTableError,
                   ParameterError,
                   NoType, StrType, NumType, FloatType, IntType, AllType,
                   Column,
                   BaseInputter, ContinuationLinesInputter,
                   BaseHeader,
                   BaseData,
                   BaseOutputter, TableOutputter,
                   BaseReader,
                   BaseSplitter, DefaultSplitter, WhitespaceSplitter,
                   convert_numpy,
                   masked
                   )
from .daophot import Daophot
from .ecsv import Ecsv
from .fastbasic import (FastBasic,
                        FastCsv,
                        FastTab,
                        FastNoHeader,
                        FastCommentedHeader,
                        FastRdb)
from .fixedwidth import (FixedWidth, FixedWidthNoHeader,
                         FixedWidthTwoLine, FixedWidthSplitter,
                         FixedWidthHeader, FixedWidthData)
from .html import HTML
from .ipac import Ipac
from .latex import Latex, AASTex, latexdicts
from .rst import RST
from .sextractor import SExtractor
from .ui import (set_guess, get_reader, read, get_writer, write, get_read_trace)

from . import connect
