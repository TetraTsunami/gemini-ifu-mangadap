from astropy.io import fits
import numpy as np

hdu = fits.open("FinalCube.fits")
# Everything in the first 500 images is garbage, so set the variance to be very high (1)
hdu[2].data[0:1000] = np.ones(hdu[2].data[0].shape)
hdu.writeto("FinalerCube.fits", overwrite=True)