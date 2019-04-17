from astropy import units as u

from numpy import pi, sqrt, allclose, count_nonzero

from .filters import filter_parameters
from .utils import get_neff



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

    
    Methods
    ----------

    psf_model
    
    calc_psf_fwhm
    
    update_effarea
    
    update_psf
        
        
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

    band1: dict
        ``{eff_wave: float, eff_width: float}``

    band2: dict
        ``{eff_wave: float, eff_width: float}``

    bandpass1: 1-d float array
        ``[eff_wave - eff_width*0.5, eff_wave+eff_width*0.5]``
    
    bandpass2: 1-d float array
        ``[eff_wave - eff_width*0.5, eff_wave+eff_width*0.5]``
    
    eff_area: float
        Effective area computed using the eff_epd size.
    
    read_noise: float
        RMS noise in the detetors per frame read
    
    Examples
    --------
    >>> duet = Telescope()
    >>> allclose(duet.eff_epd.value, 24.2)
    True
    

    """

    def __init__(self, on_axis=True):
        self.EPD = 26*u.cm
        
        if on_axis:
            self.eff_epd = 24.2*u.cm
            psf_fwhm_um = 6.7*u.micron
        else:
            self.eff_epd = 23.1*u.cm
            psf_fwhm_um = 10*u.micron

        
        pixel = 10*u.micron
        plate_scale = 6.4*u.arcsec / pixel  # arcsec per micron
        
        self.pixel = plate_scale * pixel
        
        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim. 
    
        self.read_noise = 3
    
        # Pointing jitter:
        self.psf_jitter = 5*u.arcsec

        [self.band1, self.band2] = filter_parameters()   

        center_D1 = self.band1['eff_wave'].to(u.nm).value
        width_D1 = self.band1['eff_width'].to(u.nm).value
        self.bandpass1 =[center_D1 - 0.5*width_D1, center_D1+0.5*width_D1] * u.nm

        center_D2 = self.band2['eff_wave'].to(u.nm).value
        width_D2 = self.band2['eff_width'].to(u.nm).value
        self.bandpass2 =[center_D2 - 0.5*width_D2, center_D2+0.5*width_D2] * u.nm
        

        self.psf_params = {  
            'sig':[2.08, 4.26]*u.arcsec,
            'amp':[1., 0.1]
            }


        self.update()

        
    def info(self):
        print('-----')
        print('DUET Telescope State:')
        print('Physical Entrance Pupil: {}'.format(self.EPD))
        print('Effective EPD: {}'.format(self.eff_epd))
        print('Effective Area: {}'.format(self.eff_area))        
        print('Transmission Efficiency: {}'.format(self.trans_eff))
        print()
        print('Pixel size: {}'.format(self.pixel))
        print('PSF FWHM: {}'.format(self.psf_fwhm))
        print('Pointing jitter: {}'.format(self.psf_jitter))
        print('Effective PSF FWHM: {}'.format(self.psf_size))
        print('N_eff: {}'.format(self.neff))
        print()
        print('Band 1: {}'.format(self.band1))
        print('Bandpass 1: {}'.format(self.bandpass1))
        print('Band 2: {}'.format(self.band2))
        print('Bandpass 2: {}'.format(self.bandpass2))
        print()
        print('Read noise (RMS per read): {}'.format(self.read_noise))
        print('-----')

    def update(self):
        '''
        Update paramters that are derived from other values
    
        '''      
        self.psf_fwhm = self.calc_psf_fwhm()
        self.update_psf()
        self.update_effarea()
        self.neff = get_neff(self.psf_size, self.pixel)
    


    # Allow some things to get updated.
    def calc_psf_fwhm(self):
        '''
        Computes the FWHM of the 2D PSF kernel
    
        Returns
        -------
        fwhm in astropy units
    
        '''
        pix_size = 0.01*u.arcsec
        psf_model = self.psf_model(pixel_size=pix_size)

        psf1d = (psf_model.array).sum(axis=1)
        thresh = psf1d.max() * 0.5
        above = psf1d > thresh

        return count_nonzero(above) *pix_size
    
    def update_psf(self):
        self.psf_size = sqrt(self.psf_fwhm**2 + self.psf_jitter**2)
    
    def update_effarea(self):
        self.eff_area = pi * (self.eff_epd*0.5)**2
        
    def psf_model(self, pixel_size=None, **kwargs):
        '''
        Return a astropy.convolution.Gaussian2DKernel that is the combination
        of Gaussians specified in __init__


        Other Parameters
        ----------------
        Pixel size: float
            Pixel size for the PSF kernel. Default is self.pixel
    
    
        '''        
        from astropy.convolution import Gaussian2DKernel

        if pixel_size is None:
            pixel_size = self.pixel

        set = False
        for s, n in zip(self.psf_params['sig'], self.psf_params['amp']):
            if not set:
                temp = Gaussian2DKernel( (s / pixel_size).to('').value)                
                temp_amp = temp.array.max()
                model = (n / temp_amp)*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)                
                set = True
            else:
                temp = Gaussian2DKernel( (s / pixel_size).to('').value)                
                temp_amp = temp.array.max()
                model += (n / temp_amp)*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)
                
        
        # Renorm so PSF == 1
        renorm = model.array.sum()
        set = False
        for s, n in zip(self.psf_params['sig'], self.psf_params['amp']):
            if not set:
                temp = Gaussian2DKernel( (s / pixel_size).to('').value)                
                temp_amp = temp.array.max()
                model = (n / (renorm*temp_amp))*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)     
                set = True
            else:
                temp = Gaussian2DKernel( (s / pixel_size).to('').value)                
                temp_amp = temp.array.max()
                model += (n / (temp_amp*renorm))*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)

        norm = model.array.sum()
    


        return model
    
    

