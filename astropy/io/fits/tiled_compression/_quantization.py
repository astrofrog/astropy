"""
This file contains the code for Quantizing / Dequantizing floats.
"""
import numpy as np

from astropy.io.fits.hdu.base import BITPIX2DTYPE
from astropy.io.fits.tiled_compression._compression import (
    quantize_double_c,
    quantize_float_c,
    unquantize_double_c,
    unquantize_float_c,
)

__all__ = ["Quantize"]


N_RANDOM = 10000


def _generate_random():
    # This generates a canonical list of 10000 'random' values which the FITS
    # standard requires for quantization.
    a = 16807.0;
    m = 2147483647.0;
    rand_value = np.zeros(N_RANDOM)
    seed = 1;
    for ii in range(N_RANDOM):
        temp = a * seed
        seed = temp - m * int(temp / m);
        rand_value[ii] = seed / m;
    if seed != 1_043_618_065:
        raise ValueError(f"Unexpected 10,000th seed: {seed}")
    return rand_value


RANDOM_VALUES = _generate_random()


DITHER_METHODS = {"NO_DITHER": -1, "SUBTRACTIVE_DITHER_1": 1, "SUBTRACTIVE_DITHER_2": 2}


class QuantizationFailedException(Exception):
    pass


class Quantize:
    """
    Quantization of floating-point data following the FITS standard.
    """

    def __init__(self, row: int, dither_method: int, quantize_level: int, bitpix: int):
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
        Unquantize data

        Parameters
        ----------
        buf
            The buffer to unquantize.

        Returns
        -------
        buf
            The unquantized buffer.
        """
        qbytes = np.asarray(buf)
        qbytes = qbytes.astype(qbytes.dtype.newbyteorder("="))
        # TODO: figure out if we need to support null checking
        if self.dither_method == -1:
            # For NO_DITHER we should just use the scale and zero directly
            return qbytes * scale + zero

        qbytes = qbytes.ravel()

        if self.bitpix == -32:
            output = np.zeros(qbytes.shape, dtype=np.float32)
        elif self.bitpix == -64:
            output = np.zeros(qbytes.shape, dtype=float)
        else:
            raise TypeError("bitpix should be one of -32 or -64")

        n_values = len(qbytes)

        iseed = (self.row - 1) % N_RANDOM
        nextrand = int(RANDOM_VALUES[iseed] * 500)

        # TODO: re-write the following without a loop!

        for ii in range(n_values):
            if self.dither_method == 2 and qbytes[ii] == -2147483646:
                output[ii] = 0.
            else:
                output[ii] = (qbytes[ii] - RANDOM_VALUES[nextrand] + 0.5) * scale + zero
            nextrand += 1
            if nextrand == N_RANDOM:
                iseed += 1
                if iseed == N_RANDOM:
                    iseed = 0
                nextrand = int(RANDOM_VALUES[iseed] * 500)

        return output.data

    def encode_quantized(self, buf):
        """
        Quantize data.

        Parameters
        ----------
        buf
            The buffer to quantize.

        Returns
        -------
        buf
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
            return np.frombuffer(qbytes, dtype=np.int32).data, scale, zero
