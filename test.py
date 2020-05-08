import faulthandler
faulthandler.enable()
from astropy.wcs import WCS
wcs = WCS(naxis=1)
wcs.wcs.auxprm
