import astropy.units as u
import numpy as np


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