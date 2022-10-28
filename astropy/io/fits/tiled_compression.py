"""
This module contains low level helper functions for compressing and decompressing bytes for the Tiled Table Compression algorithms as specified in the FITS 4 standard.
"""
from collections import namedtuple
from gzip import decompress as gzip_decompress


def decompress_gzip_1(cbytes: bytes):
    return gzip_decompress(cbytes)


def decompress_gzip_2(cbytes: bytes):
    raise NotImplementedError


def decompress_rice_1(cbytes: bytes, blocksize: int, bytepix: int):
    raise NotImplementedError


def decompress_plio_1(cbytes: bytes):
    raise NotImplementedError


def decompress_hcompress_1(cbytes: bytes, scale: float):
    raise NotImplementedError


def compress_gzip_1(dbytes: bytes):
    raise NotImplementedError


def compress_gzip_2(dbytes: bytes):
    raise NotImplementedError


def compress_rice_1(dbytes: bytes, blocksize: int, bytepix: int):
    raise NotImplementedError


def compress_plio_1(dbytes: bytes):
    raise NotImplementedError


def compress_hcompress_1(dbytes: bytes, scale: float):
    raise NotImplementedError


AlgorithmPair = namedtuple("AlgorithmPair", ("compression", "decompression"))


ALGORITHMS = {
    "GZIP_1": AlgorithmPair(compress_gzip_1, decompress_gzip_1),
    "GZIP_2": AlgorithmPair(compress_gzip_2, decompress_gzip_2),
    "RICE_1": AlgorithmPair(compress_rice_1, decompress_rice_1),
    "PLIO_1": AlgorithmPair(compress_plio_1, decompress_plio_1),
    "HCOMPRESS_1": AlgorithmPair(compress_hcompress_1, decompress_hcompress_1),
}


def decompress_tile(cbytes: bytes, *, algorithm: str, **kwargs):
    """
    Decompress the bytes of a tile using the given compression algorithm.

    Parameters
    ----------
    cbytes
        The compressed bytes to be decompressed.
    algorithm
        A supported decompression algorithm.
    kwargs
        Any parameters for the given compression algorithm
    """
    return ALGORITHMS[algorithm].decompression(cbytes, **kwargs)


def compress_tile(dbytes: bytes, *, algorithm: str, **kwargs):
    """
    Compress the bytes of a tile using the given compression algorithm.

    Parameters
    ----------
    dbytes
        The decompressed bytes to be compressed.
    algorithm
        A supported compression algorithm.
    kwargs
        Any parameters for the given compression algorithm
    """
    return ALGORITHMS[algorithm].compression(dbytes, **kwargs)
