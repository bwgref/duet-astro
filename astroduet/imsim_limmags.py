import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astroduet.bbmag import bb_abmag_fluence
from astroduet.image_utils import construct_image, find, ap_phot, run_daophot
from astroduet.config import Telescope
from astroduet.background import background_pixel_rate
from astroduet.utils import duet_abmag_to_fluence
from astropy.table import Table
from astropy.io import fits

def imsim_suite(**kwargs):
    """
    Simulate images of sources in galaxies with a range of surface brightnesses
    Warning! Takes several hours to run on a laptop!
    
    VERY MUCH WORK IN PROGRESS!! DO NOT USE!
    
    Parameters
    ----------   
    tel: 'min' or 'best', default is best
        Sets telescope parameters
    
    gal: 'spiral', 'elliptical' or 'dwarf', default is spiral
        Sets Sersic index and size
     
    zodi: 'low', 'med' or 'high', default is low 
        Use the medium zodiacal background rate. Overrides low_zodi.
        
    sfb: [sfb_low, sfb_high], default is [15,23] 
        List of lowest and highest surface brightness to simulate
    
    nmags: int, default is 50
        Number of source magnitudes to simulate, in 0.1 mag steps with sfb at the center, for each sfb
        
    nsrc: int, default is 100
        Number of sources to simulate at each source mag
        
    Returns
    -------
    telescope_gal_sfb_srcmag_zodi_band.fits: fits file with simulated, written to data directory 
    """
    
    # Deal with kwargs:
    tel = kwargs.pop('tel', 'best')
    gal = kwargs.pop('gal', 'spiral')
    zodi = kwargs.pop('zodi', 'low')
    sfb_lim = kwargs.pop('sfb', [18,26])
    nmags = kwargs.pop('nmags', 50)
    nsrc = kwargs.pop('nsrc', 100)

    # set some telescope, instrument parameters; check this bit with new setup files
    duet = Telescope()
    read_noise = duet.read_noise
    
    # Define image simulation parameters
    exposure = 300 * u.s
    frame = np.array([30,30]) # Dimensions of the image I'm simulating in DUET pixels (30x30 ~ 3x3 arcmin)
    psf_fwhm_pix = duet.psf_fwhm / duet.pixel
    
    # Get backgrounds
    if zodi == 'low':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, low_zodi = True, diag=False)
    elif zodi == 'med':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, med_zodi = True, diag=False)
    elif zodi == 'high':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, high_zodi = True, diag=False)
    
    # Define galaxy: Fix this and get numbers for both bands. Probably some of this needs to be done in the for loop.
    if gal == 'spiral':
        reff = 16.5 *u.arcsec
        gal_params = {'amplitude': 1,'r_eff': reff/(duet.pixel/6),'n':1, 'x_0': 0, 'y_0': 0}
    elif gal == 'elliptical':
        reff = 12.5 *u.arcsec
        gal_params = {'amplitude': 1,'r_eff': reff/(duet.pixel/6),'n':4, 'x_0': 0, 'y_0': 0}
    elif gal == 'dwarf':
        reff = 7 *u.arcsec
        gal_params = {'amplitude': 1,'r_eff': reff/(duet.pixel/6),'n':1, 'x_0': 0, 'y_0': 0}
    
    # Lots of for loops...
    sfb_arr = np.arange(sfb_lim[0],sfb_lim[1]+1.) # Now in steps of 1 mag
    ref_arr = [1,3,5,8]
    
    # First DUET1
    for sfb in sfb_arr:
        # Make reference images:
        empty_hdu = fits.PrimaryHDU()
        ref_hdu = fits.HDUList([empty_hdu])
        ref_hdu = update_header(ref_hdu, im_type='reference', zodi=zodi, gal=gal, sfb=sfb, reff=reff, 
                                band='DUET1', nframes=len(ref_arr), exptime=exposure.value)
        for nref in ref_arr:
            image = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=None,
                            sky_rate=bgd_band1, n_exp=nref)
            imhdu = fits.ImageHDU(image)
            imhdu.header['NFRAMES'] = (nref, 'Number of frames in reference image')
            ref_hdu.append(imhdu)
            
        filename = 'data/imsims/'+tel+'_duet1_'+gal+'_'+string(sfb)+'_zodi-'+zodi+'_reference.fits'
        ref_hdu.writeto(filename)
        
        # Make source images:
        srcmag_arr = np.arange(sfb - 0.5*nmags*0.1, sfb + (0.5*nmags + 1)*0.1, step=0.1) # Currently in steps of 0.1 mag
        for srcmag in srgmag_arr:
            empty_hdu = fits.PrimaryHDU()
            src_hdu = fits.HDUList([empty_hdu])
            src_hdu = update_header(src_hdu, im_type='source', zodi=zodi, gal=gal, sfb=sfb, reff=reff, 
                                    band='DUET1', srcmag=srcmag, nframes=len(ref_arr), exptime=exposure.value)
            src_fluence = duet_abmag_to_fluence(srcmag*u.ABmag, duet.bandpass1)
            for i in range(nsrc):
                source_loc = np.array([np.random.random, np.random.random]) # determine suitable source locations!
                image = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=src_fluence,
                            source_loc=source_loc, sky_rate=bgd_band1, n_exp=1)
                imhdu = fits.ImageHDU(image)
                imhdu.header['SRC_POSX'] = (source_loc[0]*len(frame[0]), 'X-position of source in image (pixels)')
                imhdu.header['SRC_POSY'] = (source_loc[1]*len(frame[1]), 'Y-position of source in image (pixels)')
                src_hdu.append(imhdu)
            
            filename = 'data/imsims/'+tel+'_duet1_'+gal+'_'+string(sfb)+'_zodi-'+zodi+'_src_'+string(srcmag)+'.fits'
            src_hdu.writeto(filename)
        
    # Get this working, then copy for DUET2
                
        
def update_header(hdu, **kwargs):
    """
    Update fits header for simulated images.
    
    Parameters
    ----------
    hdu: HDU object
        The HDU extension that you want to update

    Other keywords (default = None)
    --------------
    im_type: string
        Type of images in fits file ('reference' or 'source')
        
    zodi: string
        Setting of zodiacal background ('low', 'med', 'high')
        
    gal: string
        Galaxy morphology ('spiral', 'elliptical', 'dwarf')

    sfb: float
        Surface brightness at effective radius in ABmag/arcsec^2
        
    reff: float
        Effective radius in arcseconds
        
    srcmag: float
        Source magnitude in ABmag
               
    band: string
        DUET band ('DUET1', 'DUET2')
        
    nframes: float
        Number of image extensions
        
    exptime: float
        Exposure time in seconds
    
    Returns
    -------
    Updated HDU object
    """

    im_type = kwargs.pop('im_type', None)
    zodi = kwargs.pop('zodi', None)
    gal = kwargs.pop('gal', None)
    sfb = kwargs.pop('sfb', None)
    reff = kwargs.pop('reff', None)
    srcmag = kwargs.pop('srcmag', None)
    band = kwargs.pop('band', None)
    nframes = kwargs.pop('nframes', None)
    exptime = kwargs.pop('exptime', None)
    
    head = hdu[0].header
    head['NEXTEND'] = nframes
    head['FILETYPE'] = (im_type, 'type of images in fits file')
    head['DUETBAND'] = band
    head['EXPTIME'] = (exptime, 'Exposure time per frame (seconds)')
    head['GALTYPE'] = (gal, 'galaxy morphology')
    head['SFB_EFF'] = (sfb, 'galaxy surface brighness at R_eff (ABmag/arcsec^2)') 
    head['R_EFF'] = (reff, 'galaxy effective radius (arcsec)')
    head['ZODI_LEV'] = (zodi, 'Zodiacal background level')
    if im_type == 'source':
        head['SRC_MAG'] = (srcmag, 'source magnitude (ABmag)')
    
    return hdu
    
    
    