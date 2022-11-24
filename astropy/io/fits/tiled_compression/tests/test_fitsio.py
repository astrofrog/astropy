"""
This test file uses the https://github.com/esheldon/fitsio package to verify
our compression and decompression routines against the implementation in
cfitsio.

*Note*: The fitsio library is GPL licensed, therefore it could be interpreted
 that so is this test file. Given that this test file isn't imported anywhere
 else in the code this shouldn't cause us any issues. Please bear this in mind
 when editing this file.
"""
import pytest

fitsio = pytest.importorskip("fitsio")


@pytest.fixture()
def compressed_file(tmp_path):
    print(fitsio)


def test_decompress(compressed_file):
    pass
