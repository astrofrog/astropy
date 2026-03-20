"""
This file contains the code for Quantizing / Dequantizing floats.
"""

from functools import lru_cache

import numpy as np

from astropy.io.fits.hdu.compressed._compression import (
    quantize_double_c,
    quantize_float_c,
)

__all__ = ["Quantize"]

_N_RANDOM = 10000
_ZERO_VALUE = np.int32(-2147483646)

DITHER_METHODS = {
    "NONE": 0,
    "NO_DITHER": -1,
    "SUBTRACTIVE_DITHER_1": 1,
    "SUBTRACTIVE_DITHER_2": 2,
}


@lru_cache(maxsize=1)
def _init_randoms():
    """Initialize the array of random numbers used for dithering.

    This uses the same deterministic algorithm as CFITSIO to ensure
    bit-exact compatibility.
    """
    a, m = 16807.0, 2147483647.0
    seed = 1.0
    values = np.empty(_N_RANDOM, dtype=np.float32)
    for i in range(_N_RANDOM):
        seed = a * seed - m * int(a * seed / m)
        values[i] = seed / m
    assert int(seed) == 1043618065, "Random number sequence is incorrect"
    return values


def _dither_values(row, n):
    """Get the random dither values for n pixels at a given row."""
    rand_values = _init_randoms()
    iseed = (row - 1) % _N_RANDOM
    nextrand = int(rand_values[iseed] * 500)

    if nextrand + n <= _N_RANDOM:
        return rand_values[nextrand : nextrand + n]

    result = np.empty(n, dtype=np.float32)
    pos = 0
    while pos < n:
        chunk = min(_N_RANDOM - nextrand, n - pos)
        result[pos : pos + chunk] = rand_values[nextrand : nextrand + chunk]
        pos += chunk
        iseed = (iseed + 1) % _N_RANDOM
        nextrand = int(rand_values[iseed] * 500)
    return result


def _unquantize(data, row, scale, zero, dither_method, output_dtype):
    """Unquantize integer data to floating point using dithering.

    Parameters
    ----------
    data : ndarray
        Quantized integer data.
    row : int
        Row number, used to seed the dither sequence.
    scale : float
        BSCALE value.
    zero : float
        BZERO value.
    dither_method : int
        Dithering method (1=SUBTRACTIVE_DITHER_1, 2=SUBTRACTIVE_DITHER_2).
    output_dtype : numpy dtype
        Output data type (float32 or float64).
    """
    rand = _dither_values(row, len(data))
    output = np.subtract(data, rand, dtype=np.float64)
    output += 0.5
    output *= scale
    output += zero

    if dither_method == 2 and data.dtype == np.int32:
        output[data == _ZERO_VALUE] = 0.0

    if output.dtype != output_dtype:
        output = output.astype(output_dtype)
    return output


class QuantizationFailedException(Exception):
    pass


BITPIX2DTYPE = {-32: np.float32, -64: np.float64}


class Quantize:
    """
    Quantization of floating-point data following the FITS standard.
    """

    def __init__(
        self, *, row: int, dither_method: int, quantize_level: int, bitpix: int
    ):
        super().__init__()
        self.row = row
        # TODO: pass dither method as a string instead of int?
        self.quantize_level = quantize_level
        self.dither_method = dither_method
        self.bitpix = bitpix

    # NOTE: below we use decode_quantized and encode_quantized instead of
    # decode and encode as we need to break with the numcodec API and take/return
    # scale and zero in addition to quantized value. We should figure out how
    # to properly use the numcodec API for this use case.

    def decode_quantized(self, buf, scale, zero):
        """
        Unquantize data.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to unquantize.

        Returns
        -------
        np.ndarray
            The unquantized buffer.
        """
        data = np.asarray(buf)
        data = data.astype(data.dtype.newbyteorder("="))
        if self.dither_method == -1:
            return data * scale + zero
        if self.bitpix not in (-32, -64):
            raise TypeError("bitpix should be one of -32 or -64")
        return _unquantize(
            data.ravel(), self.row, scale, zero,
            self.dither_method, BITPIX2DTYPE[self.bitpix],
        )

    def encode_quantized(self, buf):
        """
        Quantize data.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to quantize.

        Returns
        -------
        np.ndarray
            A buffer with quantized data.
        """
        uarray = np.asarray(buf)
        uarray = uarray.astype(uarray.dtype.newbyteorder("="))
        # TODO: figure out if we need to support null checking
        if uarray.dtype.itemsize == 4:
            qbytes, status, scale, zero = quantize_float_c(
                uarray.tobytes(),
                self.row,
                uarray.size,
                1,
                0,
                0,
                self.quantize_level,
                self.dither_method,
            )[:4]
        elif uarray.dtype.itemsize == 8:
            qbytes, status, scale, zero = quantize_double_c(
                uarray.tobytes(),
                self.row,
                uarray.size,
                1,
                0,
                0,
                self.quantize_level,
                self.dither_method,
            )[:4]
        if status == 0:
            raise QuantizationFailedException()
        else:
            return np.frombuffer(qbytes, dtype=np.int32), scale, zero
