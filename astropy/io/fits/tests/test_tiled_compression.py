import numpy as np
import pytest
from numpy.testing import assert_equal

from astropy.io import fits
from astropy.io.fits.tiled_compression import compress_tile, decompress_tile
from astropy.utils.misc import NumpyRNGContext

COMPRESSION_TYPES = ["RICE_1", "PLIO_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1"]


@pytest.mark.parametrize('compression_type', COMPRESSION_TYPES)
def test_basic(tmp_path, compression_type):

    # In future can pass in settings as part of the parameterization
    settings = {}

    # Generate compressed file dynamically

    with NumpyRNGContext(42):
        original_data = np.random.randint(0, 100, 10000).reshape((100, 100)).astype('>i2')

    if compression_type == 'GZIP_2':
        settings['itemsize'] = original_data.dtype.itemsize

    header = fits.Header()

    hdu = fits.CompImageHDU(
        original_data, header, compression_type=compression_type, tile_size=(25, 25)
    )

    hdu.writeto(tmp_path / 'test.fits')

    # Load in raw compressed data
    hdulist = fits.open(tmp_path / 'test.fits', disable_image_compression=True)

    tile_shape = (hdulist[1].header['ZTILE2'], hdulist[1].header['ZTILE1'])

    # Test decompression of the first tile

    compressed_tile_bytes = hdulist[1].data['COMPRESSED_DATA'][0].tobytes()

    tile_data_bytes = decompress_tile(compressed_tile_bytes, algorithm=compression_type, **settings)

    tile_data = np.frombuffer(tile_data_bytes, dtype='>i2').reshape(tile_shape)

    assert_equal(tile_data, original_data[:25, :25])

    # Now compress the original data and compare to compressed bytes. Since
    # the exact compressed bytes might not match (e.g. for GZIP it will depend
    # on the compression level) we instead put the compressed bytes into the
    # original BinTableHDU, then read it in as a normal compressed HDU and make
    # sure the final data match.

    compressed_tile_bytes = compress_tile(original_data[:25, :25].tobytes(), algorithm=compression_type, **settings)

    hdulist[1].data['COMPRESSED_DATA'][0] = np.frombuffer(compressed_tile_bytes, dtype=np.uint8)
    hdulist[1].writeto(tmp_path / 'updated.fits')
    hdulist.close()

    hdulist_new = fits.open(tmp_path / 'updated.fits')
    assert_equal(hdulist_new[1].data, original_data)
    hdulist_new.close()
