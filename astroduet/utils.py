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
    
def galex_to_duet(galmags):
    """
    Converts GALEX FUV and NUV ABmags into DUET 1 and DUET 2 ABmags, assuming flat Fnu


    Parameters
    ----------
    galmags: array
        GALEX AB magnitudes, either as [[FUV1, ..., FUVN],[NUV1, ..., NUVN]] or as [[FUV1, NUV1],...,[FUVN, NUVN]]
        Code assumes the first format if len(galmags) = 2        
    
    Returns
    -------
    duetmags: Array with same shape as galmags, with DUET 1 and DUET 2 ABmags.

    Example 
    -------
    >>> galmags = [20,20]
    >>> duetmags = galex_to_duet(galmags)
    >>> np.allclose(duetmags, [20,20])
    True

    """

    import astroduet.config as config
    from astropy.modeling.blackbody import FNU
    
    # Setup filters (only interested in effective wavelengths/frequency)
    duet = config.Telescope()
    
    galex_fuv_lef = 151.6 * u.nm
    galex_nuv_lef = 226.7 * u.nm
    
    duet_1_lef = duet.band1['eff_wave']
    duet_2_lef = duet.band2['eff_wave']
    
    galex_fuv_nef = galex_fuv_lef.to(u.Hz, u.spectral())
    galex_nuv_nef = galex_nuv_lef.to(u.Hz, u.spectral())
    
    duet_1_nef = duet_1_lef.to(u.Hz, u.spectral())
    duet_2_nef = duet_2_lef.to(u.Hz, u.spectral())
    
    # Sort input array into FUV and NUV magnitudes
    if len(galmags) == 2:
        fuv_mag = galmags[0]*u.ABmag
        nuv_mag = galmags[1]*u.ABmag
    else:
        fuv_mag = galmags[:,0]*u.ABmag
        nuv_mag = galmags[:,1]*u.ABmag
        
    # Convert GALEX magnitudes to flux densities
    fuv_fnu = fuv_mag.to(FNU, u.spectral_density(galex_fuv_nef))
    nuv_fnu = nuv_mag.to(FNU, u.spectral_density(galex_nuv_nef))
    
    # Extrapolate to DUET bands assuming linear Fnu/nu
    delta_fnu = (nuv_fnu - fuv_fnu)/(galex_nuv_nef - galex_fuv_nef)
    
    d1_fnu = fuv_fnu + delta_fnu*(duet_1_nef - galex_fuv_nef)
    d2_fnu = fuv_fnu + delta_fnu*(duet_2_nef - galex_fuv_nef)
    
    # Convert back to magnitudes
    d1_mag = d1_fnu.to(u.ABmag, u.spectral_density(duet_1_nef))
    d2_mag = d2_fnu.to(u.ABmag, u.spectral_density(duet_2_nef))
    
    # Construct output array
    if len(galmags) == 2:
        duetmags = np.array([d1_mag.value, d2_mag.value])
    else:
        duetmags = np.array([d1_mag.value, d2_mag.value]).transpose()
    
    return duetmags



