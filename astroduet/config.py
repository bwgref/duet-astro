from astropy import units as u
import numpy as np
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
    config : string
        Defualts to 'baseline'.
        If in the list of available options, sets the duet parameters to this version.


    Returns
    ----------
    An initialized instance of the Telescope class.


    Methods
    ----------

    info

    apply_filters

    fluence_to_rate
    
    rate_to_fluence
    
    calc_snr
    
    psf_model

    calc_psf_hpd

    compute_psf_norms

    calc_radial_profile

    update_effarea

    update_psf_vals

    update_bandpass

    _set_baseline

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

    plate_scale : float
        Astropy units value for arcsec / micron

    Examples
    --------
    >>> duet = Telescope()
    >>> allclose(duet.eff_epd.value, 24.2)
    True


    """

    def __init__(self, config='baseline'):

        self.config_list = ['baseline', 'classic', 'minimum_mass', 
            'fine_plate','equal_mass','largest_aperture', 'reduced_baseline']

        assert config in self.config_list, 'Bad config option "'+config+'"'
            
        if config is 'baseline':
            self.set_baseline()
        elif config is 'reduced_baseline':
            self.set_reduced_baseline()
        elif config is 'classic':
            self.set_classic()
        elif config is 'minimum_mass':
            self.set_minimum_mass()
        elif config is 'fine_plate':
            self.set_fine_plate()
        elif config is 'equal_mass':
            self.set_equal_mass()
        elif config is 'largest_aperture':
            self.set_largest_aperature()
        
        self.config=config
            
        # Detector-specific values
        # Set QE files here:
        self.qe_files = {
            'description' : ['DUET 1 CBE QE', 'DUET 2 CBE QE'],
            'names' : [datadir+'detector_180_220nm.csv', datadir+'detector_260_300nm.csv']
        }

        # Dark current given in e- per pixel per sec
        self.dark_current_downscale = 4
        self.dark_current = (0.046 / self.dark_current_downscale) * u.ph / u.s

        # RMS value
        self.read_noise = 7

        # Pointing jitter:
        self.psf_jitter = 5*u.arcsec

        # Compute the effective area
        self.update_effarea()

        # Below just adds the pointing jitter in quadrature to the PSF FWHM.
        # Not really used for anything...
        self.update_psf()

        # Compute the effective number of background pixels (this isn't used in as
        # many places and should be depricated moving forward.
        self.neff = get_neff(self.psf_size, self.pixel)

        # Compute the filters (to be hard coded in the futuer?)
        [self.band1, self.band2] = filter_parameters(duet=self)
        center_D1 = self.band1['eff_wave'].to(u.nm).value
        width_D1 = self.band1['eff_width'].to(u.nm).value
        self.bandpass1 =[center_D1 - 0.5*width_D1, center_D1+0.5*width_D1] * u.nm

        center_D2 = self.band2['eff_wave'].to(u.nm).value
        width_D2 = self.band2['eff_width'].to(u.nm).value
        self.bandpass2 =[center_D2 - 0.5*width_D2, center_D2+0.5*width_D2] * u.nm

    def set_baseline(self):
        '''Baseline configuration. Duplicate this with different values
        and/or 
        '''
    
        self.EPD = 26*u.cm
        self.eff_epd = 24.2*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 6.4*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim.

        # Below are in 
        self.psf_params = {
        'sig':[2.08, 4.26]*u.arcsec,
        'amp':[1, 0.1],
        'norm':[0.505053538858156, 0.21185072119504136]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 5.0 * u.arcsec

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'al_mgf2_mirror_coatings.csv'
        }

        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }
        
    def set_classic(self):
        '''Baseline configuration. Duplicate this with different values
        and/or 
        '''
    
        self.EPD = 26*u.cm
        self.eff_epd = 24.5*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 6.25*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim.

        # Below are in 
        self.psf_params = {
        'sig':[3.2*u.micron*self.plate_scale],
        'amp':[1.0],
        'norm':[0.624335784627981]
        }

        # Computed by calc_psf_fwhm, but hardcoded here for speed.
        self.psf_fwhm = 4.6 * u.arcsec

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'al_mgf2_mirror_coatings.csv'
        }

        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }
        
    def set_minimum_mass(self):
        '''Minimum mass configuration. 
        
        '''
    
        self.EPD = 26*u.cm
        self.eff_epd = 24.5*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 6.67*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel
        self.jitter_rms = 11.8 * u.micron * self.plate_scale

        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim.

        self.psf_params = {
        'sig':[5.3*u.micron*self.plate_scale],
        'amp':[1.0],
        'norm':[0.9845499186721847]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 8.2 * u.arcsec

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'al_mgf2_mirror_coatings.csv'
        }

        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }

    def set_fine_plate(self):
        '''Fine plate scale configuration. 
        
        '''
    
        self.EPD = 26*u.cm
        self.eff_epd = 23.9*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 5.03*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim.

        self.psf_params = {
        'sig':[2.8*u.micron*self.plate_scale],
        'amp':[1],
        'norm':[0.48927044820341287]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 3.2 * u.arcsec

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'al_mgf2_mirror_coatings.csv'
        }

        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }

    def set_equal_mass(self):
        '''Largest aperture configuration. 
        
        '''
    
        self.EPD = 31.9*u.cm
        self.eff_epd = 30.0*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 5.03*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim.

        self.psf_params = {
        'sig':[4.8*u.micron*self.plate_scale],
        'amp':[1],
        'norm':[0.9589514449942292]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 5.4 * u.arcsec

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'al_mgf2_mirror_coatings.csv'
        }

        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }
        
    def set_largest_aperature(self):
        '''Largest aperture configuration. 
        
        '''
    
        self.EPD = 33.8*u.cm
        self.eff_epd = 32.0*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 5.03*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim.

        self.psf_params = {
        'sig':[6.0*u.micron*self.plate_scale],
        'amp':[1.0],
        'norm':[0.9967376175195871]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 7.0 * u.arcsec

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'al_mgf2_mirror_coatings.csv'
        }

        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }
        
    def set_reduced_baseline(self):
        '''Reduced baseline configuration. Duplicate this with different values
        and/or 
        '''
        
        reduction = 0.8
        self.EPD = 26*u.cm
        self.eff_epd = reduction*24.2*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 6.4*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        # Transmission through the Schmidt plates
        self.trans_eff = (0.975)**8 # from Jim.

        self.psf_params = {
        'sig':[2.08, 4.26]*u.arcsec,
        'amp':[1, 0.1],
        'norm':[0.505053538858156, 0.21185072119504136]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 5.0 * u.arcsec
        
        # Set QE files here:
        
        self.qe_files = {
            'description' : ['DUET 1 CBE QE', 'DUET 2 CBE QE'],
            'names' : [datadir+'detector_180_220nm.csv', datadir+'detector_260_300nm.csv']
        }

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'al_mgf2_mirror_coatings.csv'
        }

        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_filter_light.csv', datadir+'duet2_filter_light.csv']
        }

    def info(self):
        info_str = f"""-----
        DUET Telescope State: {self.config}
        Physical Entrance Pupil: {self.EPD}
        Effective EPD: {self.eff_epd}
        Effective Area: {self.eff_area}
        Transmission Efficiency: {self.trans_eff}
        
        Pixel size: {self.pixel}
        PSF FWHM: {self.psf_fwhm}
        Pointing jitter: {self.psf_jitter}
        Effective PSF FWHM: {self.psf_size}
        N_eff: {self.neff}

        Band 1: {self.band1}
        Bandpass 1: {self.bandpass1}
        Band 2: {self.band2}
        Bandpass 2: {self.bandpass2}

        Dark current: {self.dark_current}
        Read noise (RMS per read): {self.read_noise}
        -----
        """
        print(info_str)
        return info_str

    def update_bandpass(self):
        '''
        Update bandpass values based on whatever set of files are stores in


        '''

        [self.band1, self.band2] = filter_parameters(duet=self)

        center_D1 = self.band1['eff_wave'].to(u.nm).value
        width_D1 = self.band1['eff_width'].to(u.nm).value
        self.bandpass1 =[center_D1 - 0.5*width_D1, center_D1+0.5*width_D1] * u.nm

        center_D2 = self.band2['eff_wave'].to(u.nm).value
        width_D2 = self.band2['eff_width'].to(u.nm).value
        self.bandpass2 =[center_D2 - 0.5*width_D2, center_D2+0.5*width_D2] * u.nm

    def update_psf_vals(self):
        '''
        Update paramters that are derived from other values.

        This needs to still re-compute the PSF normalizations at some point, but
        that's not implemented here.

        '''
        self.psf_params['norm'] = self.compute_psf_norms()
        self.psf_fwhm = self.calc_psf_fwhm()
        self.update_psf()
        self.neff = get_neff(self.psf_size, self.pixel)

    def calc_radial_profile(self):
        '''
        The python way, from Stack Overflow
        https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile

        Returns the radial profile and the pixel size used to computer. 

        '''
        import numpy as np
        pix_size = 0.01*u.arcsec
        xsize = 2001
        ysize = 2001
        psf_model = self.psf_model(pixel_size=pix_size, x_size=xsize, y_size=ysize)
        data = psf_model.array

        center = [(xsize*0.5), (ysize*0.5)]
        y,x = np.indices((data.shape)) # first determine radii of all pixels
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

    def fluence_to_rate(self, fluence):
        '''
        Helper script to convert fluences to count rates

        '''

        rate = self.eff_area * self.trans_eff * fluence
        return rate

    def rate_to_fluence(self, rate):
        '''Convert count rates to fluences.'''

        fluence = rate / (self.eff_area * self.trans_eff)
        return fluence

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

    def apply_filters(self, wave, spec, band=1, **kwargs):
        """

        Applies the reflectivity, QE, and red-filter based on the input files
        in the data subdirectory. See the individual scripts or set diag=True
        to see what filters are beign used.

        Parameters
        ----------
        wave : float array
            The array containing the wavelengths of the spcetrum

        spec : float array
            The spectrum that you want to filter.

        Other parameters
        ----------------

        band : int
            Use band 1 (default) or band 2 files


        Returns
        -------
        band_flux : float array
            The spectrum after the filtering has been applied. Will have the
            same length as "spec".


        Examples
        --------
        >>> from astroduet.config import Telescope
        >>> duet = Telescope()
        >>> wave = [190, 200]*u.nm
        >>> spec = [1, 1]
        >>> band_flux = duet.apply_filters(wave, spec, band=1)
        >>> test = [0.20659143, 0.37176641]
        >>> allclose(band_flux, test)
        True

        """

        from astroduet.filters import load_reflectivity, load_qe, load_redfilter, apply_trans

        # Shift to make band an index:

        band_ind = band - 1
        qe_file = self.qe_files['names'][band_ind]
        reflectivity_file = self.reflectivity_file['name']
        bandpass_file = self.bandpass_files['names'][band_ind]

        # Load filters
        ref_wave, reflectivity = load_reflectivity(infile = reflectivity_file, **kwargs)
        qe_wave, qe = load_qe(infile = qe_file, **kwargs)
        red_wave, red_trans = load_redfilter(infile = bandpass_file, **kwargs)

        # Apply filters
        ref_flux = apply_trans(wave, spec, ref_wave, reflectivity)
        qe_flux = apply_trans(wave, ref_flux, qe_wave, qe)
        band_flux = apply_trans(wave, qe_flux, red_wave, red_trans)

        return band_flux

    def calc_snr(self, texp, src_rate, bgd_rate, nint = 1.0):
        """

        Compute the signal-to-noise ratio for a given exposure, source rate,
        and background rate. Have this in Telescope() because this also depends on
        the number of effective background pixels and the read noise, both of
        which are attributes to duet.

        Parameters
        ----------
        texp : float
            Exposure time

        src_rate : float array
            Source rate in photons per second.

        bgd_rate : float array
            Background rate in photons per second

        Other parameters
        ----------------

        nint : int
            How many exposures you want to stack (Default is 1).


        Returns
        -------
        snr : float array
            Same size as src_rate. SNR for each entry


        """
        src_rate = src_rate.to(u.ph/u.s).value
        bgd_rate = bgd_rate.to(u.ph/u.s).value
        texp = texp.to(u.s).value
        denom = (nint*src_rate*texp +
            nint * self.neff * (bgd_rate*texp + self.read_noise**2))**0.5
        nom = nint*src_rate * texp
        snr = nom / denom
        return snr

