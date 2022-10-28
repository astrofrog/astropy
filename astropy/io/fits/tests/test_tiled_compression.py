import numpy as np
import pytest
from numpy.testing import assert_equal

from astropy.io import fits
from astropy.io.fits.tiled_compression import decompress_tile
from astropy.utils.misc import NumpyRNGContext

COMPRESSION_TYPES = ["RICE_1", "PLIO_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1"]


@pytest.mark.parametrize('compression_type', COMPRESSION_TYPES)
def test_basic(tmp_path, compression_type):

    # Generate compressed file dynamically

    with NumpyRNGContext(42):
        original_data = np.random.randint(0, 100, 10000).reshape((100, 100)).astype('>i2')

    header = fits.Header()

    hdu = fits.CompImageHDU(
        original_data, header, compression_type=compression_type, tile_size=(25, 25)
    )

    hdu.writeto(tmp_path / 'test.fits')

    # Load in raw compressed data
    hdulist = fits.open(tmp_path / 'test.fits', disable_image_compression=True)

    tile_shape = (hdulist[1].header['ZTILE2'], hdulist[1].header['ZTILE1'])

    # Test the first tile

    compressed_tile_bytes = hdulist[1].data['COMPRESSED_DATA'][0].tobytes()

    tile_data_bytes = decompress_tile(compressed_tile_bytes, algorithm=compression_type)

    tile_data = np.frombuffer(tile_data_bytes, dtype='>i2').reshape(tile_shape)

    assert_equal(tile_data, original_data[:25, :25])
