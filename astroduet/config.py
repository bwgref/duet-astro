from astropy import units as u
from numpy import pi, sqrt, allclose

class Telescope():
    """
    Make a Telescope object containing various instrument parameters
    
    Parameters
    ----------
    on_axis : conditional, default True
        Use the on-axis effective area and PSF sizes or use the 
        edge of the field of view values.


    Returns
    ----------
    An initialized instance of the Telescope class.



    Attributes
    ----------
    epd: float
        The physical size of the entrance pupil.

    eff_epd: float
        The effective size of the entrance pupil (i.e. once corrected
        for vignetting, with Astropy units.

    psf_fwhm: float
        The FWHM of the PSF, with Astropy units.
                
    psf_jitter: float
        The contribution of the PSF to be added in quadrature with the psf_fwhm
        due to spacecraft pointing jitter.
        
    psf_size: float
        psf_fwhm and psf_jitter added in quadrature with Astropy units.
        
    pixel: float
        Angular pixel size with Astropy units

    bandone: 1-d float array
        ``[eff_band, eff_width]``

    bandtwo: 1-d float array
        ``[eff_band, eff_width]``
    
    eff_area: float
        Effective area computed using the eff_epd size.
    
    Examples
    --------
    >>> duet = Telescope()
    >>> allclose(duet.eff_epd.value, 24.2)
    True
    

    """

    def __init__(self, on_axis=True):
        self.EPD = 30*u.cm
        
        if on_axis:
            self.eff_epd = 24.2*u.cm
            psf_fwhm_um = 6.7*u.micron
        else:
            self.eff_epd = 23.1*u.cm
            psf_fwhm_um = 10*u.micron

        
        pixel = 10*u.micron
        plate_scale = 6.4*u.arcsec / pixel  # arcsec per micron
        
        self.psf_fwhm = plate_scale * psf_fwhm_um
        self.pixel = plate_scale * pixel
        
        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim. 
    
        # Pointing jitter:
        self.psf_jitter = 5*u.arcsec
        self.update_psf()
        self.update_effarea()


        center_D1 = 208
        width_D1 = 53
        self.bandone=[center_D1 - 0.5*width_D1, center_D1+0.5*width_D1]*u.nm

        center_D2 = 284
        width_D2 = 68
        self.bandtwo=[center_D2 - 0.5*width_D2, center_D2+0.5*width_D2]*u.nm
        
        
    def info(self):
        print('Physical Entrance Pupil: {}'.format(self.EPD))
        print('Effective EPD: {:5.3}'.format(self.eff_epd))
        print('Effective Area: {:8.3}'.format(self.eff_area))        
        print('Pixel size: {:5.2}'.format(self.pixel))
        print('Transmission Efficiency: {:5.2}'.format(self.trans_eff))
        print('PSF FWHM: {:5.2}'.format(self.psf_fwhm))
        print()
        print('Pointing jitter: {}'.format(self.psf_jitter))
        self.update_psf()
        print('Effective PSF FWHM: {:0.2}'.format(self.psf_size))
        print()
        print('Band 1: {}'.format(self.bandone))
        print('Band 2: {}'.format(self.bandtwo))
        
    # Allow some things to get updated.
    def update_psf(self):
        self.psf_size = sqrt(self.psf_fwhm**2 + self.psf_jitter**2)
    
    def update_effarea(self):
        self.eff_area = pi * (self.eff_epd*0.5)**2
