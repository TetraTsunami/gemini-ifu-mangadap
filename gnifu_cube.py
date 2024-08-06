from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

from mangadap.datacube.datacube import DataCube
from mangadap.util.sampling import Resample, angstroms_per_pixel

class GNIFUCube(DataCube):
    r"""
    Container class for a custom datacube.

    Args:
        ifile (:obj:`str`):
            The name of the file to read.
    """

    instrument = 'gnifu'
    """
    Set the name of the instrument.  This is used to construct the
    output file names.
    """

    def __init__(self, ifile):

        _ifile = Path(ifile).resolve()
        if not _ifile.exists():
            raise FileNotFoundError(f'File does not exist: {_ifile}')

        # Set the paths
        self.directory_path = _ifile.parent
        self.file_name = _ifile.name

        # Collect the metadata into a dictionary
        meta = {}
        meta['z'] = 0.0245 # Specific to MRK-203. Taken from the MaNGA survey (thanks!)
        sres = 2454.8387 # Calculated with Christy. I forget how we got this, but it's what I have written down.
        
        with fits.open(str(_ifile)) as hdu:
            print('Reading datacube ...', end='\r')
            prihdr = hdu[1].header
            wcs = WCS(header=prihdr, fix=True)

            flux = hdu[1].data.T
            #ivar = np.ma.power(hdu[2].data.T, -1.)
            err = np.ma.sqrt(hdu[2].data.T)
            mask = np.ma.getmaskarray(err).copy()
            err = err.filled(0.0)

        print("Reading datacube ... It's so over")
        
        nwave = flux.shape[-1]
        spatial_shape = flux.shape[:-1]
        coo = np.array([np.ones(nwave), np.ones(nwave), np.arange(nwave)+1]).T
        wave = wcs.all_pix2world(coo, 1)[:,2] # *wcs.wcs.cunit[2].to('angstrom')        
        # - Convert the fluxes to flux density
        dw = angstroms_per_pixel(wave, regular=False)
        flux /= dw[None,None,:]
        # - Set the geometric step to the mean value.  This means some
        # pixels will be oversampled and others will be averaged.
        dlogl = np.mean(np.diff(np.log10(wave)))
        # - Resample all the spectra.  Note that the Resample arguments
        # expect the input spectra to be provided in 2D arrays with the
        # last axis as the dispersion axis.
        r = Resample(flux.reshape(-1,nwave), e=err.reshape(-1,nwave),
                     mask=mask.reshape(-1,nwave), x=wave, inLog=False, newRange=wave[[0,-1]],
                     newLog=True, newdx=dlogl)
        # - Reshape and reformat the resampled data in prep for
        # instantiation
        ivar = r.oute.reshape(*spatial_shape,-1)
        mask = r.outf.reshape(*spatial_shape,-1) < 0.8
        ivar[mask] = 0.0
        gpm = np.logical_not(mask)
        ivar[gpm] = 1/ivar[gpm]**2
        _sres = np.full(ivar.shape, sres, dtype=float)
        flux = r.outy.reshape(*spatial_shape,-1)

        # Instantiate the base class
        super().__init__(flux, ivar=ivar, mask=mask,
                         sres=_sres, wave=r.outx, meta=meta,
                        #  sres=sres, wave=None, meta=meta,
                         prihdr=prihdr, wcs=None, name=_ifile.name.split('.')[0])
        
if __name__ == '__main__':
    cube = GNIFUCube("FinalCube.fits")
    print(cube)
    
# manga_dap -f FinalerCube.fits --cube_module gnifu_cube GNIFUCube --plan_module mangadap.config.analysisplan.AnalysisPlan -p default_plan.toml -o dap