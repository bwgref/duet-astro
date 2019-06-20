from astropy import units as u
import numpy as np
from numpy import pi, sqrt, allclose, count_nonzero
import warnings

from .filters import filter_parameters
from .utils import get_neff
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.psf import EPSFModel

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

    construct_epsf

    Attributes
    ----------
    epd : float
        The physical size of the entrance pupil.

    eff_epd : float
        The effective size of the entrance pupil (i.e. once corrected
        for vignetting, with Astropy units.

    pixel : float
        Angular pixel size with Astropy units

    band1 : dict
        ``{eff_wave: float, eff_width: float}``

    band2 : dict
        ``{eff_wave: float, eff_width: float}``

    bandpass1 : 1-d float array
        ``[eff_wave - eff_width*0.5, eff_wave+eff_width*0.5]``

    bandpass2 : 1-d float array
        ``[eff_wave - eff_width*0.5, eff_wave+eff_width*0.5]``

    eff_area : float
        Effective area computed using the eff_epd size.

    filter_shift : [shift_1, shift_2], both quantities indicating length
        Shift the redfilter windows by this amount. E.g. [0 * u.nm, 10 * u.nm]

    read_noise : float
        RMS noise in the detetors per frame read

    plate_scale : float
        Astropy units value for arcsec / micron

    pointing_rms : float
        Astropy units for PSF blur associated with the pointing instability.
        Given in arcseconds.

    psf_fwhm : float
        The FWHM of the PSF, with Astropy units. Pre-computed by calc_psf_fwhm().

    epsf_model : class 'photutils.psf.models.EPSFModel'
        2D fittable model for use by photutils.psf.DAOPhotPSFPhotometry

    Examples
    --------
    >>> duet = Telescope()
    >>> allclose(duet.eff_epd.value, 24.5)
    True


    """

    def __init__(self, config='minimum_mass'):

        self.config_list = ['baseline', 'classic', 'minimum_mass',
            'fine_plate','equal_mass','largest_aperture', 'reduced_baseline']

        assert config in self.config_list, 'Bad config option "'+config+'"'

        self.filter_shift = [0 * u.nm, 0 * u.nm]

        if config == 'baseline':
            self.set_baseline()
        elif config == 'reduced_baseline':
            self.set_reduced_baseline()
        elif config == 'classic':
            self.set_classic()
        elif config == 'minimum_mass':
            self.set_minimum_mass()
        elif config == 'fine_plate':
            self.set_fine_plate()
        elif config == 'equal_mass':
            self.set_equal_mass()
        elif config == 'largest_aperture':
            self.set_largest_aperature()

        self.config=config

        # Detector-specific values
        # Set QE files here:
        self.qe_files = {
            'description' : ['DUET 1 CBE QE', 'DUET 2 CBE QE'],
            'names' : [datadir+'duet1_qe_20190518_v2.csv',
                       datadir+'duet2_qe_20190518_v3.csv']
        }

        self.transmission_file = datadir+'glass_transmission_20190518.csv'

        # Bandpass files here
        self.bandpass_files = {
            'description' : ['CBE DUET 1 Bandpass', 'CBE DUET 2 Bandpass'],
            'names' : [datadir+'duet1_bandpass_20190518.csv',
                       datadir+'duet2_bandpass_20190518.csv']
        }

        self.reflectivity_file = {
            'description' : 'CBE Reflectivity',
            'name' : datadir+'mirror_coatings_20190518.csv'
        }

        # Dark current given in e- per pixel per sec
        # Updated on 2019/06/03 end-of-life numbers at 0.5 krad from Rick
        self.dark_current = 1.4e-3 * u.ph / u.s # electrons / pixel / sec @ 190K
        # RMS value
        self.read_noise = 4 # e- RMS per read.


        # Pointing jitter:
        self.pointing_rms = 2.5*u.arcsec

        # Compute the effective area
        self.update_effarea()

        # Compute the effective number of background pixels (this isn't used in as
        # many places and should be depricated moving forward.
        self.neff = get_neff(self.psf_fwhm, self.pixel)

        # Compute the filters (to be hard coded in the futuer?)
        [self.band1, self.band2] = filter_parameters(duet=self)
        center_D1 = self.band1['eff_wave'].to(u.nm).value
        width_D1 = self.band1['eff_width'].to(u.nm).value
        self.bandpass1 =[center_D1 - 0.5*width_D1, center_D1+0.5*width_D1] * u.nm

        center_D2 = self.band2['eff_wave'].to(u.nm).value
        width_D2 = self.band2['eff_width'].to(u.nm).value
        self.bandpass2 =[center_D2 - 0.5*width_D2, center_D2+0.5*width_D2] * u.nm

        # Construct fittable ePSF model
        self.construct_epsf()

    def shift_filters(self, filter_shift):
        """Shift the redfilter by this amount.

        Parameters
        ----------
        filter_shift : [shift_1, shift_2], both quantities indicating length
            Shift the redfilter windows by this amount. E.g. [0 * u.nm, 10 * u.nm]

        """

        self.filter_shift = filter_shift
        self.update_bandpass()

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

        # Below are in
        self.psf_params = {
        'sig':[2.08, 4.26]*u.arcsec,
        'amp':[1, 0.1]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 10.5 * u.arcsec

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


        # Below are in
        self.psf_params = {
        'sig':[3.2*u.micron*self.plate_scale],
        'amp':[1.0]
        }

        # Computed by calc_psf_fwhm, but hardcoded here for speed.
        self.psf_fwhm = 10.0 * u.arcsec

    def set_minimum_mass(self):
        '''Minimum mass configuration.

        '''

        self.EPD = 26*u.cm
        self.eff_epd = 24.5*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 6.56*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        # From Jason, DUET1, field- and band-average CBE PSF 1-sigma:
        self.psf_params = {
        'sig':[4.25*u.arcsec],
        'amp':[1.0]
        }

        # Includes pointing jitter, computed by hand adding
        # above FWHM (13-arcsec) in quadrature with a 2.5-arcsec rms
        # pointing jitter
        self.psf_fwhm = 11.5 * u.arcsec

    def set_fine_plate(self):
        '''Fine plate scale configuration.

        '''

        self.EPD = 26*u.cm
        self.eff_epd = 23.9*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 5.03*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel

        self.psf_params = {
        'sig':[2.8*u.micron*self.plate_scale],
        'amp':[1]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 8.5 * u.arcsec

    def set_equal_mass(self):
        '''Largest aperture configuration.

        '''

        self.EPD = 31.9*u.cm
        self.eff_epd = 30.0*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 5.03*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel


        self.psf_params = {
        'sig':[4.8*u.micron*self.plate_scale],
        'amp':[1]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 9.5 * u.arcsec


    def set_largest_aperature(self):
        '''Largest aperture configuration.

        '''

        self.EPD = 33.8*u.cm
        self.eff_epd = 32.0*u.cm
        psf_fwhm_um = 6.7*u.micron
        pixel = 10*u.micron
        self.plate_scale = 5.03*u.arcsec / pixel  # arcsec per micron
        self.pixel = self.plate_scale * pixel
        
        self.psf_params = {
        'sig':[6.0*u.micron*self.plate_scale],
        'amp':[1.0]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 10.5 * u.arcsec


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

        self.psf_params = {
        'sig':[2.08, 4.26]*u.arcsec,
        'amp':[1, 0.1]
        }
        # Computed by calc_psf_hpd, but hardcoded here.
        self.psf_fwhm = 10.5 * u.arcsec



    def info(self):
        info_str = f"""-----
        DUET Telescope State: {self.config}
        Physical Entrance Pupil: {self.EPD}
        Effective EPD: {self.eff_epd}
        Effective Area: {self.eff_area}
        
        Pixel size: {self.pixel}
        Pointing RMS: {self.pointing_rms}
        Effective PSF FWHM: {self.psf_fwhm}
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

    def update_bandpass(self, **kwargs):
        '''
        Update bandpass values based on whatever set of files are stores in


        '''

        [self.band1, self.band2] = filter_parameters(duet=self, **kwargs)

        center_D1 = self.band1['eff_wave'].to(u.nm).value
        width_D1 = self.band1['eff_width'].to(u.nm).value
        self.bandpass1 =[center_D1 - 0.5*width_D1, center_D1+0.5*width_D1] * u.nm

        center_D2 = self.band2['eff_wave'].to(u.nm).value
        width_D2 = self.band2['eff_width'].to(u.nm).value
        self.bandpass2 =[center_D2 - 0.5*width_D2, center_D2+0.5*width_D2] * u.nm

    def calc_radial_profile(self, pix_size = 0.1*u.arcsec):
        '''
        The python way, from Stack Overflow
        https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile

        Returns the radial profile and the pixel size used to compute the PSF FWHM.

        '''
        import numpy as np

        # Cover +/- 25 arcsec
        cover = 25 * u.arcsec
        nbins = np.floor(( 2 * (cover / pix_size).value)) + 1
        psf_model = self.psf_model(pixel_size=pix_size, x_size = nbins, y_size = nbins)
        data = psf_model.array

        center = [(nbins*0.5), (nbins*0.5)]
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

    def update_effarea(self):
        self.eff_area = pi * (self.eff_epd*0.5)**2

    def fluence_to_rate(self, fluence):
        '''
        Helper script to convert fluences to count rates

        '''
#        rate = self.eff_area * self.trans_eff * fluence
        rate = self.eff_area * fluence

        return rate

    def rate_to_fluence(self, rate):
        '''Convert count rates to fluences.'''

#        fluence = rate / (self.eff_area * self.trans_eff)
        fluence = rate / (self.eff_area )

        return fluence

    def psf_model(self, pixel_size=None, **kwargs):
        '''
        Return a astropy.convolution.Gaussian2DKernel that is the combination
        of Gaussians specified in __init__


        Other Parameters
        ----------------
        Pixel size: float
            Pixel size for the PSF kernel. Default is self.pixel

        Accepts options keywords for astropy.convlution.Gaussian2DKernel
        '''

        if pixel_size is None:
            pixel_size = self.pixel
            force_renorm=False
        else:
            force_renorm = True

        for i, (s, n) in enumerate(zip(self.psf_params['sig'], self.psf_params['amp'])):
            if i == 0:
                psf_model = n * Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)
            else:
                psf_model +=  n * Gaussian2DKernel( (s / pixel_size).to('').value, **kwargs)
                force_renorm = True

        if force_renorm is True:
            psf_model.normalize()

        # Step 2: Add pointing jitter:
        pointing_rms = self.pointing_rms

        pointing_model = Gaussian2DKernel( (pointing_rms / pixel_size).to('').value, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Ignore warning for doctest
            model = convolve(psf_model, pointing_model)
        model.normalize()

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
        >>> test = [0.27842462, 0.29616992]
        >>> allclose(band_flux, test)
        True
        """

        from astroduet.filters import load_reflectivity, load_qe, \
            load_redfilter, apply_trans, load_transmission

        # Shift to make band an index:

        band_ind = band - 1
        qe_file = self.qe_files['names'][band_ind]
        reflectivity_file = self.reflectivity_file['name']
        bandpass_file = self.bandpass_files['names'][band_ind]

        # Load filters
        ref_wave, reflectivity = load_reflectivity(infile = reflectivity_file, **kwargs)
        qe_wave, qe = load_qe(infile = qe_file, **kwargs)
        red_wave, red_trans = \
            load_redfilter(infile = bandpass_file,
                           shift_by=self.filter_shift[band_ind], **kwargs)
        trans_wave, transmission = \
            load_transmission(infile=self.transmission_file, **kwargs)

        # Apply filters
        ref_flux = apply_trans(wave, spec, ref_wave, reflectivity)
        qe_flux = apply_trans(wave, ref_flux, qe_wave, qe)
        trans_flux = apply_trans(wave, qe_flux, trans_wave, transmission)
        band_flux = apply_trans(wave, trans_flux, red_wave, red_trans)
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

    def construct_epsf(self):
        """

        Build the integrated PSF fittable model used by DAOPhot
        and store it in self.epsf_model.

        """
        # For best operation, kernel needs to be oversampled but only minimally
        oversample = 2

        # Get kernel - array size must be odd for convolution purposes
        psf_os = self.psf_model(pixel_size=self.pixel/oversample,
                                x_size=15, y_size=15)
        self.epsf_model = EPSFModel(psf_os.array, oversampling=oversample,
                                    normalization_correction=1/oversample**2)

