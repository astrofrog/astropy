"""
This module contains low level helper functions for compressing and decompressing bytes for the Tiled Table Compression algorithms as specified in the FITS 4 standard.
"""
from collections import namedtuple
from gzip import decompress as gzip_decompress


def decompress_gzip_1(cbytes: bytes) -> bytes:
    """
    Decompress bytes using the GZIP_1 algorithm.

    The Gzip algorithm is used in the free GNU software compression utility of
    the same name. It was created by J.- L. Gailly and M. Adler, based on the
    DEFLATE algorithm (Deutsch 1996), which is a combination of LZ77 (Ziv &
    Lempel 1977) and Huffman coding.

    Parameters
    ----------
    cbytes
        The bytes to decompress.

    Returns
    -------
    dbytes
        The decompressed bytes.
    """
    return gzip_decompress(cbytes)


def decompress_gzip_2(cbytes: bytes) -> bytes:
    """
    Decompress bytes using the GZIP_2 algorithm.

    The gzip2 algorithm is a variation on ’GZIP 1’. In this case the bytes in
    the array of data values are shuffled so that they are arranged in order of
    decreasing significance before being compressed.

    For example, a five-element contiguous array of two-byte (16-bit) integer
    values, with an original big-endian byte order of:

    .. math::
        A1 A2 B1 B2 C1 C2 D1 D2 E1 E2

    will have the following byte order after shuffling:

    .. math::
        A1 B1 C1 D1 E1 A2 B2 C2 D2 E2,

    where A1, B1, C1, D1, and E1 are the most-significant bytes from
    each of the integer values.

    Byte shuffling shall only be performed for integer or floating-point
    numeric data types; logical, bit, and character types must not be shuffled.

    Parameters
    ----------
    cbytes
        The bytes to decompress.

    Returns
    -------
    dbytes
        The decompressed bytes.
    """
    raise NotImplementedError


def decompress_rice_1(cbytes: bytes, blocksize: int, bytepix: int) -> bytes:
    """
    Decompress bytes using the RICE_1 algorithm.

    The Rice algorithm [1]_ is simple and very fast It requires only enough
    memory to hold a single block of 16 or 32 pixels at a time. It codes the
    pixels in small blocks and so is able to adapt very quickly to changes in
    the input image statistics (e.g., Rice has no problem handling cosmic rays,
    bright stars, saturated pixels, etc.).

    References
    ----------
    .. [1] Rice, R. F., Yeh, P.-S., and Miller, W. H. 1993, in Proc. of the 9th
    AIAA Computing in Aerospace Conf., AIAA-93-4541-CP, American Institute of
    Aeronautics and Astronautics [https://doi.org/10.2514/6.1993-4541]

    Parameters
    ----------
    cbytes
        The bytes to decompress.

    blocksize
        The blocksize to use, each tile is coded into blocks a number of pixels
        wide. The default value in FITS headers is 32 pixels per block.

    bytepix
        The number of 8-bit bytes in each original integer pixel value.

    Returns
    -------
    dbytes
        The decompressed bytes.
    """
    raise NotImplementedError


def decompress_plio_1(cbytes: bytes) -> bytes:
    """
    Decompress bytes using the PLIO_1 algorithm.

    The IRAF PLIO (pixel list) algorithm was developed to store integer-valued
    image masks in a compressed form. Such masks often have large regions of
    constant value hence are highly compressible. The compression algorithm
    used is based on run-length encoding, with the ability to dynamically
    follow level changes in the image, allowing a 16-bit encoding to be used
    regardless of the image depth.

    Parameters
    ----------
    cbytes
        The bytes to decompress.

    Returns
    -------
    dbytes
        The decompressed bytes.
    """
    raise NotImplementedError


def decompress_hcompress_1(cbytes: bytes, scale: float, smooth: bool) -> bytes:
    """
    Decompress bytes using the HCOMPRESS_1 algorithm.

    Hcompress is an the image compression package written by Richard L. White
    for use at the Space Telescope Science Institute. Hcompress was used to
    compress the STScI Digitized Sky Survey and has also been used to compress
    the preview images in the Hubble Data Archive.

    The technique gives very good compression for astronomical images and is
    relatively fast. The calculations are carried out using integer arithmetic
    and are entirely reversible. Consequently, the program can be used for
    either lossy or lossless compression, with no special approach needed for
    the lossless case.

    Parameters
    ----------
    cbytes
        The bytes to decompress.

    scale
        The integer scale parameter determines the amount of compression. Scale
        = 0 or 1 leads to lossless compression, i.e. the decompressed image has
        exactly the same pixel values as the original image. If the scale
        factor is greater than 1 then the compression is lossy: the
        decompressed image will not be exactly the same as the original

    smooth
        At high compressions factors the decompressed image begins to appear
        blocky because of the way information is discarded. This blockiness
        ness is greatly reduced, producing more pleasing images, if the image
        is smoothed slightly during decompression.

    Returns
    -------
    dbytes
        The decompressed bytes.
    """
    raise NotImplementedError


def compress_gzip_1(dbytes: bytes) -> bytes:
    """
    """
    raise NotImplementedError


def compress_gzip_2(dbytes: bytes) -> bytes:
    """
    """
    raise NotImplementedError


def compress_rice_1(dbytes: bytes, blocksize: int, bytepix: int) -> bytes:
    """
    """
    raise NotImplementedError


def compress_plio_1(dbytes: bytes) -> bytes:
    """
    """
    raise NotImplementedError


def compress_hcompress_1(dbytes: bytes, scale: float) -> bytes:
    """
    """
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
