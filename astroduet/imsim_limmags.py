import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astroduet.bbmag import bb_abmag_fluence
from astroduet.image_utils import construct_image, find, ap_phot, run_daophot
from astroduet.config import Telescope
from astroduet.background import background_pixel_rate
from astroduet.utils import duet_abmag_to_fluence
from astropy.table import Table

def limmags_diff_sb(**kwargs):
    """
    Simulate difference imaging of new sources in galaxies with a range of surface brightnesses
    Warning! Takes several hours to run on a laptop!
    
    VERY MUCH WORK IN PROGRESS!! DO NOT USE!
    
    Parameters
    ----------    
    low_zodi: conditional, default is True 
        Use the low zodiacal background rate

    med_zodi: conditional, default is False 
        Use the medium zodiacal background rate. Overrides low_zodi.

    high_zodi: conditional, default is False 
        Use the medium zodiacal background rate. Overrideslow_zodi.
        
    sfb: [sfb_low, sfb_high], default is [15,23] 
        List of lowest and highest surface brightness to simulate
    
    nmags: int, default is 50
        Number of source magnitudes to simulate, in 0.1 mag steps with sfb at the center, for each sfb
        
    nsrc: int, default is 100
        Number of sources to simulate at each source mag
        
    Returns
    -------
    limmags_diff_sb_zodi.fits: fits table with results, written to data directory 
        Columns: SFB, srcmag, src_rate_in_D1, src_det_D1, med_src_rate_psf_D1, med_src_rate_psf_D1_err, 
        src_rate_in_D2, src_det_D2, med_src_rate_psf_D2, med_src_rate_psf_D2_err        
    """
    
    # Deal with kwargs:
    low_zodi = kwargs.pop('low_zodi', True)
    med_zodi = kwargs.pop('med_zodi', False)
    high_zodi = kwargs.pop('high_zodi', False)
    sfb = kwargs.pop('sfb', [15,23])
    nmags = kwargs.pop('nmags', 50)
    nsrc = kwargs.pop('nsrc', 100)

    # set some telescope, instrument parameters
    duet = Telescope()
    read_noise = duet.read_noise
    
#    [bgd_band1, bgd_band2] = background_pixel_rate(duet, low_zodi = low_zodi, med_zodi, diag=True)
    
    
    
    # Define image simulation parameters
    exposure = 300 * u.s
    frame = np.array([30,30]) # Dimensions of the image I'm simulating in DUET pixels (30x30 ~ 3x3 arcmin)
    psf_fwhm_pix = duet.psf_fwhm / duet.pixel