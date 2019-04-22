from astropy import units as u

from numpy import pi, sqrt, allclose, count_nonzero

from .filters import filter_parameters
from .utils import get_neff
from astropy.convolution import Gaussian2DKernel

import os
curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'data')+'/'


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
    
    calc_psf_hpd
    
    calc_radial_profile
    
    update_effarea
    
    update_psf_vals
    
        
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
            'amp':[1, 0.1],
            'norm':[0.505053538858156, 0.21185072119504136]
            }


        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 5.0 * u.arcsec

        # Compute the effective area
        self.update_effarea()

        # Below just adds the pointing jitter in quadrature
        self.update_psf()
        
        # Compute the effective number of background pixels (this isn't used in as
        # many places and should be depricated moving forward.
        self.neff = get_neff(self.psf_size, self.pixel)

        self.qe_files = {
            'description' : ['DUET 1 CBE QE', 'DUET 2 CBE QE'],
            'names' : [datadir+'detector_180_220nm.csv', datadir+'detector_260_300nm.csv']
        }
        
        self.reflectivity_files = {
            'description' : ['CBE Reflectivity'],
            'names' : [datadir+'al_mgf2_mirror_coatings.csv']
        }

        
        self.bandpass_filter_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],            
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }


        
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

    def update_psf_vals(self):
        '''
        Update paramters that are derived from other values.
        
        This needs to still re-compute the PSF normalizations at some point, but
        that's not implemented here.
    
        '''
        self.psf_params['norm'] = self.compute_psf_norms()
        self.psf_fwhm = self.calc_psf_hpd()
        self.update_psf()
        self.neff = get_neff(self.psf_size, self.pixel)

    
    def calc_radial_profile(self):
        '''
        The python way, from Stack Overflow
        https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile        

        Returns the radial profile and the pixel size

        '''
        import numpy as np
        pix_size = 0.1*u.arcsec
        xsize = 1001
        ysize = 1001
        psf_model = self.psf_model(pixel_size=pix_size, x_size=xsize, y_size=ysize)
        data = psf_model.array

        center = [(xsize*0.5), (ysize*0.5)]
        y,x = indices((data.shape)) # first determine radii of all pixels
        r = np.sqrt((x-center[0])**2+(y-center[1])**2)
        ind = np.argsort(r.flat) # get sorted indices
        sr = r.flat[ind] # sorted radii
        sim = data.flat[ind] # image values sorted by radii
        ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
        # determining distance between changes
        deltar = ri[1:] - ri[:-1] # assume all radii represented
        rind = np.where(deltar)[0] # location of changed radius
        nr = rind[1:] - rind[:-1] # number in radius bin
        csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
        tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
        radialprofile = tbin/nr # the answer
        
        return pix_size, np.array(radialprofile)


    # Allow some things to get updated.
    def calc_psf_fwhm(self):
        '''
        Computes the FWHM of the 2D PSF kernel. Only do this if you change the PSF
        parameters because it takes a long time!
    
        Returns
        -------
        fwhm in astropy units
    
        '''

        pix_size, rad_profile = self.calc_radial_profile()
        thresh = rad_profile.max() * 0.5
        above = (rad_profile > thresh)
        fwhm = 2.0*(count_nonzero(above)*pix_size)
        return fwhm
    
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
        if pixel_size is None:
            pixel_size = self.pixel
            force_renorm=False
        else:
            force_renorm = True

        if not force_renorm:
            set = False
            for s, n in zip(self.psf_params['sig'], self.psf_params['norm']):
                if not set:
                    model = n*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)
                    set=True
                else:
                    model += n*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)
        else:
            norms = self.compute_psf_norms(pixel_size=pixel_size)
            set= False
            for s, n in zip(self.psf_params['sig'], norms):
                if not set:
                    model = n*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)
                    set=True
                else:
                    model += n*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)

        return model
    
    
    
    def compute_psf_norms(self, pixel_size=None, **kwargs):
        """
        Helper script to convert PSF amplitudes to PSF normalizations    
    
        Prints out the modified normalizations that you should copy up into
        the Telescope class definition above.

        Other Parameters
        ----------------
    
        Pixel size: float
            Pixel size for the PSF kernel. Default is Telscope().pixel
 
        Returns
        -------
        
        [norm1, norm2, ...] proper normalization of the Gaussian given the pixel size.
 
        """


        if pixel_size is None:
            pixel_size = self.pixel

        diag = kwargs.pop('diag', False)

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
        new_norms = []
        for ind, [s, n] in enumerate(zip(self.psf_params['sig'], self.psf_params['amp'])):
            temp = Gaussian2DKernel( (s / pixel_size).to('').value)                
            temp_amp = temp.array.max()
            new_norm = (n / (renorm*temp_amp))
            new_norms.append(new_norm)
            if not set:
                model = new_norm*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)     
                set = True
            else:
                model += new_norm*Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)

        norm = model.array.sum()
        
        if diag:
            print('Kernel normalization after PSFs renormalized {}'.format(norm)) 


        return new_norms

        

    
    
