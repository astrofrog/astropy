"""
This module contains low level helper functions for compressing and decompressing buffer for the Tiled Table Compression algorithms as specified in the FITS 4 standard.
"""
from abc import abstractmethod
from gzip import compress as gzip_compress
from gzip import decompress as gzip_decompress

import numpy as np

from astropy.io.fits.tiled_compression._compression import compress_plio_1_c, decompress_plio_1_c

__all__ = ['Gzip1', 'Gzip2', 'Rice1', 'PLIO1', 'HCompress', 'compress_tile', 'decompress_tile']


# We define our compression classes in the form of a numcodecs class. We make
# the dependency on numcodecs optional as we can use them internally without it
# (for now).
try:
    from numcodecs.abc import Codec
except ImportError:
    class Codec:
        codec_id = None
        """Codec identifier."""

        @abstractmethod
        def encode(self, buf):  # pragma: no cover
            """Encode data in `buf`.
            Parameters
            ----------
            buf : buffer-like
                Data to be encoded. May be any object supporting the new-style
                buffer protocol.
            Returns
            -------
            enc : buffer-like
                Encoded data. May be any object supporting the new-style buffer
                protocol.
            """

        @abstractmethod
        def decode(self, buf, out=None):  # pragma: no cover
            """Decode data in `buf`.
            Parameters
            ----------
            buf : buffer-like
                Encoded data. May be any object supporting the new-style buffer
                protocol.
            out : buffer-like, optional
                Writeable buffer to store decoded data. N.B. if provided, this buffer must
                be exactly the right size to store the decoded data.
            Returns
            -------
            dec : buffer-like
                Decoded data. May be any object supporting the new-style
                buffer protocol.
            """


class Gzip1(Codec):
    """
    The FITS GZIP 1 compression and decompression algorithm.

    The Gzip algorithm is used in the free GNU software compression utility of
    the same name. It was created by J. L. Gailly and M. Adler, based on the
    DEFLATE algorithm (Deutsch 1996), which is a combination of LZ77 (Ziv &
    Lempel 1977) and Huffman coding.
    """
    codec_id = "FITS_GZIP1"

    def decode(self, buf):
        """
        Decompress buffer using the GZIP_1 algorithm.

        {_GZIP_1_DESCRIPTION}

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            The decompressed buffer.
        """
        return gzip_decompress(bytes(buf))

    def encode(self, buf):
        """
        Compress the data in the buffer using the GZIP_1 algorithm.

        Parameters
        ----------
        buf
            The buffer to compress.

        Returns
        -------
        buf
            A buffer with compressed data.
        """
        return gzip_compress(bytes(buf))


class Gzip2(Codec):
    """
    The FTIS GZIP2 compression and decompression algorithm.

    The gzip2 algorithm is a variation on 'GZIP 1'. In this case the buffer in
    the array of data values are shuffled so that they are arranged in order of
    decreasing significance before being compressed.

    For example, a five-element contiguous array of two-byte (16-bit) integer
    values, with an original big-endian byte order of:

    .. math::
        A1 A2 B1 B2 C1 C2 D1 D2 E1 E2

    will have the following byte order after shuffling:

    .. math::
        A1 B1 C1 D1 E1 A2 B2 C2 D2 E2,

    where A1, B1, C1, D1, and E1 are the most-significant buffer from
    each of the integer values.

    Byte shuffling shall only be performed for integer or floating-point
    numeric data types; logical, bit, and character types must not be shuffled.

    Parameters
    ----------
    itemsize
        The number of buffer per value (e.g. 2 for a 16-bit integer)

    """
    codec_id = "FITS_GZIP2"

    def __init__(self, itemsize: int):
        super().__init__()
        self.itemsize = itemsize

    def decode(self, buf):
        """
        Decompress buffer using the GZIP_2 algorithm.

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            The decompressed buffer.
        """
        # Start off by unshuffling buffer
        shuffled_buffer = gzip_decompress(bytes(buf))
        array = np.frombuffer(shuffled_buffer, dtype=np.uint8)
        return array.reshape((self.itemsize, -1)).T.ravel().tobytes()

    def encode(self, buf):
        """
        Compress the data in the buffer using the GZIP_2 algorithm.

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            The decompressed buffer.
        """
        # Start off by shuffling buffer
        array = np.frombuffer(buf, dtype=np.uint8)
        shuffled_buffer = array.reshape((-1, self.itemsize)).T.ravel().tobytes()
        return gzip_compress(shuffled_buffer)


class Rice1(Codec):
    """
    The FTIS RICE1 compression and decompression algorithm.

    The Rice algorithm [1]_ is simple and very fast It requires only enough
    memory to hold a single block of 16 or 32 pixels at a time. It codes the
    pixels in small blocks and so is able to adapt very quickly to changes in
    the input image statistics (e.g., Rice has no problem handling cosmic rays,
    bright stars, saturated pixels, etc.).

    Parameters
    ----------
    blocksize
        The blocksize to use, each tile is coded into blocks a number of pixels
        wide. The default value in FITS headers is 32 pixels per block.

    bytepix
        The number of 8-bit buffer in each original integer pixel value.

    References
    ----------
    .. [1] Rice, R. F., Yeh, P.-S., and Miller, W. H. 1993, in Proc. of the 9th
    AIAA Computing in Aerospace Conf., AIAA-93-4541-CP, American Institute of
    Aeronautics and Astronautics [https://doi.org/10.2514/6.1993-4541]
    """
    codec_id = "FITS_RICE1"

    def __init__(self, blocksize: int, bytepix: int):
        self.blocksize = blocksize
        self.bytepix = bytepix

    def decode(self, buf):
        """
        Decompress buffer using the RICE_1 algorithm.

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            The decompressed buffer.
        """
        raise NotImplementedError

    def encode(self, buf):
        """
        Compress the data in the buffer using the RICE_1 algorithm.

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            A buffer with decompressed data.
        """
        raise NotImplementedError


class PLIO1(Codec):
    """
    The FTIS PLIO1 compression and decompression algorithm.

    The IRAF PLIO (pixel list) algorithm was developed to store integer-valued
    image masks in a compressed form. Such masks often have large regions of
    constant value hence are highly compressible. The compression algorithm
    used is based on run-length encoding, with the ability to dynamically
    follow level changes in the image, allowing a 16-bit encoding to be used
    regardless of the image depth.
    """
    codec_id = "FITS_PLIO1"

    def decode(self, buf):
        """
        Decompress buffer using the PLIO_1 algorithm.

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            The decompressed buffer.
        """
        cbytes = np.frombuffer(buf, dtype=np.uint8).tobytes()
        return decompress_plio_1_c(cbytes)

    def encode(self, buf):
        """
        Compress the data in the buffer using the PLIO_1 algorithm.
        """
        dbytes = np.frombuffer(buf, dtype=np.uint8).tobytes()
        return compress_plio_1_c(dbytes)


class HCompress(Codec):
    """
    The FTIS PLIO1 compression and decompression algorithm.

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
    """
    codec_id = "FITS_HCOMPRESS1"

    def __init__(self, scale: float, smooth: bool):
        self.scale = scale
        self.smooth = smooth

    def decode(self, buf):
        """
        Decompress buffer using the HCOMPRESS_1 algorithm.

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            A buffer with decompressed data.
        """
        raise NotImplementedError

    def encode(self, buf):
        """
        Compress the data in the buffer using the HCOMPRESS_1 algorithm.

        Parameters
        ----------
        buf
            The buffer to decompress.

        Returns
        -------
        buf
            A buffer with decompressed data.
        """
        raise NotImplementedError


ALGORITHMS = {
    "GZIP_1": Gzip1,
    "GZIP_2": Gzip2,
    "RICE_1": Rice1,
    "PLIO_1": PLIO1,
    "HCOMPRESS_1": HCompress,
}


def decompress_tile(buf, *, algorithm: str, **kwargs):
    """
    Decompress the buffer of a tile using the given compression algorithm.

    Parameters
    ----------
    buf
        The compressed buffer to be decompressed.
    algorithm
        A supported decompression algorithm.
    kwargs
        Any parameters for the given compression algorithm
    """
    return ALGORITHMS[algorithm](**kwargs).decode(buf)


def compress_tile(buf, *, algorithm: str, **kwargs):
    """
    Compress the buffer of a tile using the given compression algorithm.

    Parameters
    ----------
    buf
        The decompressed buffer to be compressed.
    algorithm
        A supported compression algorithm.
    kwargs
        Any parameters for the given compression algorithm
    """
    return ALGORITHMS[algorithm](**kwargs).encode(buf)
