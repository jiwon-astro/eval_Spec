"""I/O helpers."""
from astropy.io import fits

#Read spectral fits file
def open_fits(path,order=0,dtype='int'):
    hdulist = fits.open(path)
    data = hdulist[order].data
    if dtype is not None:
        data = data.astype(dtype)
    return hdulist[order].header, data
