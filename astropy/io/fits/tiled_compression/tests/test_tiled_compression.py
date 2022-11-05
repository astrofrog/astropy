import numpy as np
import pytest
from numpy.testing import assert_equal

from astropy.io import fits
from astropy.io.fits.tiled_compression import compress_tile, decompress_tile

COMPRESSION_TYPES = [
    "GZIP_1",
    "GZIP_2",
    "RICE_1",
    "HCOMPRESS_1",
    "PLIO_1",
]


@pytest.mark.parametrize('compression_type', COMPRESSION_TYPES)
def test_basic(tmp_path, compression_type):

    # In future can pass in settings as part of the parameterization
    settings = {}

    # Generate compressed file dynamically

    original_data = np.arange(144).reshape((12, 12)).astype('>i2')

    header = fits.Header()

    hdu = fits.CompImageHDU(
        original_data, header, compression_type=compression_type, tile_size=(4, 4)
    )

    hdu.writeto(tmp_path / 'test.fits')

    # Load in raw compressed data
    hdulist = fits.open(tmp_path / 'test.fits', disable_image_compression=True)

    tile_shape = (hdulist[1].header['ZTILE2'], hdulist[1].header['ZTILE1'])

    if compression_type == 'GZIP_2':
        settings['itemsize'] = original_data.dtype.itemsize
    elif compression_type == 'PLIO_1':
        settings['tilesize'] = np.product(tile_shape)
    elif compression_type == 'RICE_1':
        settings['blocksize'] = hdulist[1].header['ZVAL1']
        settings['bytepix'] = hdulist[1].header['ZVAL2']
        settings['tilesize'] = np.product(tile_shape)
    elif compression_type == 'HCOMPRESS_1':
        # TODO: generalize bytepix, we need to pick 4 or 8 and then cast down
        # later to smaller ints if needed.
        settings['bytepix'] = 4
        settings['scale'] = hdulist[1].header['ZVAL1']
        settings['smooth'] = hdulist[1].header['ZVAL2']
        settings['nx'] = hdulist[1].header['ZTILE2']
        settings['ny'] = hdulist[1].header['ZTILE1']

    # Test decompression of the first tile

    compressed_tile_bytes = hdulist[1].data['COMPRESSED_DATA'][0].tobytes()

    tile_data_bytes = decompress_tile(compressed_tile_bytes, algorithm=compression_type, **settings)

    if compression_type == 'PLIO_1':
        # In the case of PLIO_1, the bytes are always returned as 32-bit
        # native endian bits, which might differ from ZBITPIX.
        tile_data_bytes = tile_data_bytes[:np.product(tile_shape) * 4]
        tile_data = np.frombuffer(tile_data_bytes, dtype='i4').astype('>i2').reshape(tile_shape)
    elif compression_type == 'RICE_1':
        tile_data = np.frombuffer(tile_data_bytes, dtype=f'i{settings["bytepix"]}').astype('>i2').reshape(tile_shape)
    elif compression_type == 'HCOMPRESS_1':
        tile_data = np.frombuffer(tile_data_bytes, dtype='i4').astype('>i2').reshape(tile_shape)
    else:
        tile_data = np.frombuffer(tile_data_bytes, dtype='>i2').reshape(tile_shape)

    assert_equal(tile_data, original_data[:4, :4])

    # Now compress the original data and compare to compressed bytes. Since
    # the exact compressed bytes might not match (e.g. for GZIP it will depend
    # on the compression level) we instead put the compressed bytes into the
    # original BinTableHDU, then read it in as a normal compressed HDU and make
    # sure the final data match.

    if compression_type == 'PLIO_1':
        # PLIO expects specifically 32-bit ints as input - again need to find
        # a way to not special case this here.
        tile_data_bytes = original_data[:4, :4].astype('i4').tobytes()
    elif compression_type == 'RICE_1':
        # PLIO expects specifically little endian ints as input - again need to find
        # a way to not special case this here.
        tile_data_bytes = original_data[:4, :4].astype(f'i{settings["bytepix"]}').tobytes()
    elif compression_type == 'HCOMPRESS_1':
        # HCOMPRESS expects specifically little endian ints as input - again need to find
        # a way to not special case this here.
        tile_data_bytes = original_data[:4, :4].astype('i4').tobytes()
    else:
        tile_data_bytes = original_data[:4, :4].tobytes()

    compressed_tile_bytes = compress_tile(tile_data_bytes, algorithm=compression_type, **settings)

    # Then check that it also round-trips if we go through fits.open
    if compression_type == 'PLIO_1':
        hdulist[1].data['COMPRESSED_DATA'][0] = np.frombuffer(compressed_tile_bytes, dtype=np.int16)
    else:
        hdulist[1].data['COMPRESSED_DATA'][0] = np.frombuffer(compressed_tile_bytes, dtype=np.uint8)
    hdulist[1].writeto(tmp_path / 'updated.fits')
    hdulist.close()
    hdulist_new = fits.open(tmp_path / 'updated.fits')
    assert_equal(hdulist_new[1].data, original_data)
    hdulist_new.close()
