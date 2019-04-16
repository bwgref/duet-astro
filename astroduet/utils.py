import astropy.units as u
import numpy as np
import os

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'data')


def duet_abmag_to_fluence(ABmag, band, **kwargs):
    """
    Convert AB magnitude for a source into the number of source counts.


    Parameters
    ----------
    ABmag: float
        AB magnitude in the bandpass that you're using
        
    bandpass: array
        DUET bandpass you're using
        
    Returns
    -------
    Fluence in the band (ph / cm2 / sec)


    Example 
    -------
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> fluence = duet_abmag_to_fluence(20*u.ABmag, duet.bandpass1)
    >>> np.isclose(fluence.value, 0.014425401676001228)
    True

    """

    from astropy.modeling.blackbody import FLAM
    
    import numpy as np


    funit = u.ph / u.cm**2/u.s / u.Angstrom # Spectral radiances per Hz or per angstrom

    bandpass = np.abs( (band[1] - band[0])).to(u.AA)
    midband = np.mean( (band).to(u.AA) )

    fluence = bandpass *  ABmag.to(funit, equivalencies=u.spectral_density(midband))


    return fluence
    
    
def load_neff():
    """
    Load number of effective background pixels in the PSF from
    file provided by Rick Cook.

    ----
    Returns

    oversample is the ration of the PSF to the pixel size.
    neff is is the resulting value.

    """
    import os
    from numpy import genfromtxt
    ref_file = os.path.join(datadir, 'neff_data.dat')
    header=True
    neff = {}
    oversample, neff = genfromtxt(ref_file, unpack=True, skip_header=True)
    return oversample, neff


def get_neff(psf_size, pixel_size):
    """
    Determine the number of effective background pixels based on the PSF size and the
    pixel size. Assume these are given with astropy units:
    
    Parameters
    ----------
    psf_size: float
        PSF FWHM size
        
    pixel-size: float
        Physical size of pixel (in the same units as psf_size)
        
    Returns
    -------
    The effective number of background pixels that will contribute. Note this is
    fairly idealized, so it's really here as a legacy term.
 
    
     Example 
    -------
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> neff = get_neff(duet.psf_size, duet.pixel)
    >>> np.isclose(neff, 8.019141843089937)
    True

    """
    from numpy import interp
    over, neff = load_neff()
    data_oversample = (psf_size / pixel_size).value

    neff = interp(data_oversample, over, neff)
    return neff


