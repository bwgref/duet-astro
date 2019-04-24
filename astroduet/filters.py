import os
from numpy import allclose
import astropy.units as u

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'data')


def load_qe(infile = None, **kwargs):
    """
    
    Parameters
    ----------
    
    infile 
        The full path to the input QE file
    
    
    Loads the detector QE and returns the values.
    
    Returns
    -------
    wave : 1D array
        Wavelength values from teh input file, with Astropy units
        
    qe : 1D array
        QE values, normalized to 1
        
    Example
    -------
    
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> band = 1
    >>> wave, qe = load_qe(infile=duet.qe_files['names'][band])
    >>> allclose(qe[50], 0.602565)
    True
    

    """
    import numpy as np

    band = kwargs.pop('band', 1)
    diag = kwargs.pop('diag', False)

    assert infile is not None, 'load_qe: Provide an input QE file'

# 
#     if band == 1:
#         infile = os.path.join(datadir, 'detector_180_220nm.csv')
#     elif band == 2:
#         infile = os.path.join(datadir, 'detector_260_300nm.csv')
#     elif band == 3:
#         infile = os.path.join(datadir, 'detector_340_380nm.csv')
#     else:
#         raise ValueError('band number not recognized')

    f = open(infile, 'r')
    header = True
    qe = {}
    set = False
    for line in f:
        if header:
            header = False
            continue
        fields = line.split(',')
        if not set:
            wave = float(fields[0])
            qe = float(fields[3])
            set = True
        else:
            wave = np.append(wave, float(fields[0]))
            qe = np.append(qe, float(fields[3]))

    f.close()

    # Give wavelength a unit
    wave *= u.nm

    if diag:
        print('Detector Q.E. loader')
        print('Band {} has input file {}'.format(band, infile))


    return wave, qe / 100.

def load_reflectivity(infile = None, **kwargs):
    """
    Loads the optics reflectivity and returns the values.

    Parameters
    ----------
    
    infile 
        The full path to the input reflectivity file
    
    
    Loads the primary mirror reflectivity and returns the values.
    
    Returns
    -------
    wave : 1D array
        Wavelength values from teh input file, with Astropy units
        
    reflectivity : 1D array
        Reflectivity values, normalized to 1
        
    Example
    -------
    
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> wave, reflectivity = load_reflectivity(infile=duet.reflectivity_file['name'])
    >>> allclose(reflectivity[50], 0.896282)
    True

    """
    import astropy.units as ur
    import numpy as np

    diag = kwargs.pop('diag', False)
    
    assert infile is not None, 'load_reflectivity: Need an input file'
    
    
#    infile = os.path.join(datadir, 'al_mgf2_mirror_coatings.csv')

    f = open(infile, 'r')
    header = True
    qe = {}
    set = False
    for line in f:
        if header:
            header = False
            continue
        fields = line.split(',')
        if not set:
            wave = float(fields[0])
            reflectivity = float(fields[1])
            set = True
        else:
            wave = np.append(wave, float(fields[0]))
            reflectivity = np.append(reflectivity, float(fields[1]))

    f.close()

    # Give wavelength a unit
    wave *= ur.nm

    if diag:
        print('Optics reflectivity loader')
        print('Input file {}'.format(infile))


    return wave, reflectivity/100.

def load_redfilter(infile = None, **kwargs):
    """
    Loads the bandpass filter and returns the transmission values.

    Parameters
    ----------
    
    infile 
        The full path to the input bandpass file
    
        
    Returns
    -------
    wave : 1D array
        Wavelength values from the input file, with Astropy units
        
    reflectivity : 1D array
        Reflectivity values, normalized to 1
        
    Example
    -------
    
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> wave, redfilter = load_redfilter(infile=duet.bandpass_files['names'][0])
    >>> allclose(redfilter[30], 0.635886)
    True

    """
    import astropy.units as ur
    import numpy as np

    band = kwargs.pop('band', 1)
    diag = kwargs.pop('diag', False)
    light = kwargs.pop('light', True)


    assert infile is not None, 'load_redfilter: Need an input file'


    f = open(infile, 'r')
    header = True
    qe = {}
    set = False
    for line in f:
        if header:
            if (line.startswith('Wavelength')) or ('%T' in line):
                header = False
            continue
        fields = line.split(',')
        if not set:
            wave = float(fields[0])
            transmission = float(fields[1])
            set = True
        else:
            wave = np.append(wave, float(fields[0]))
            transmission = np.append(transmission, float(fields[1]))

    f.close()

    # Give wavelength a unit
    wave *= ur.nm

    if diag:
        print('Red filter loader')
        print('Band {} has input file {}'.format(band, infile))


    return wave, transmission / 100.


def apply_trans(wav_s, flux_s, wav_t, trans, **kwargs):
    """
    Interpolates transmission curve onto source wavelength array, applies transmission curve.

    Required inputs:
    wav_s (wavelength array of source spectrum, numpy array with units)
    flux_s (source spectrum, numpy array with units)
    wav_t (wavelength array of transmission curve, numpy array with units)
    trans (transmission curve, numpy array without units)

    Optional Inputs (defaults):
    diag = Diagnostics toggle (False)

    Returns corrected input spectrum

    """

    import astropy.units as ur
    import astropy.constants as cr
    import numpy as np

    # Check if the inputs make sense
    assert len(wav_s) == len(flux_s), "Input spectrum and corresponding wavelength array are not the same length"
    assert len(wav_s) == len(flux_s), "Input spectrum and corresponding wavelength array are not the same length"
    assert len(wav_t) == len(trans), "Transmission curve and corresponding wavelength array are not the same length"
    assert max(trans) <= 1., "Values larger than 1 found in transmission curve"
    assert min(trans) >= 0., "Values smaller than 0 found in transmission curve"
    assert wav_s.unit.is_equivalent(ur.m), "Input wavelength array does not have unit of length"
    assert wav_t.unit.is_equivalent(ur.m), "Transmission wavelength array does not have unit of length"

    # Make sure wav_s and wav_t are in the same units for interpolation
    wav_ttrue = wav_t.to(wav_s.unit)

    # Interpolate transmission curve onto source wavelength array
    trans_int = np.interp(wav_s,wav_ttrue,trans)

    # Correct input spectrum:
    flux_corr = flux_s * trans_int

    diag = kwargs.pop('diag', False)

    if diag:
        print('Diagnostics?')

    return flux_corr

# def apply_filters(wave, spec, qe_file=None,
#         reflectivity_file=None, bandpass_file = None,
#         **kwargs):
#     """
#     
#     Applies the reflectivity, QE, and red-filter based on the input files
#     in the data subdirectory. See the individual scripts or set diag=True
#     to see what filters are beign used.
#     
#     Parameters
#     ----------
#     wave : float array
#         The array containing the wavelengths of the spcetrum
#         
#     spec : float array
#         The spectrum that you want to filter.
# 
#     qe_file : string
#         Pathname to the QE file that you qant to use.
#         
#     reflectivity_file : string 
#         Pathname to the QE file that you qant to use.
# 
#     
#     
#     Optional Parameters
#     ----------
#     band : int
#         Which band to use. Default is band=1, but this is set (poorly) in
#         the lower-level scripts, so it's al little opaque here.
#         Can be 1 or 2.    
#     
#     
#          
#     Returns
#     -------
#     band_flux : float array
#         The spectrum after the filtering has been applied. Will have the
#         same length as "spec".
# 
#     
#     Examples
#     --------
#     >>> from astroduet.config import Telescope
#     >>> duet = Telescope()
#     >>> wave = [190, 200]*u.nm
#     >>> spec = [1, 1]
#     >>> band_flux = apply_filters(wave, spec, \
#         qe_file = duet.qe_files['names'][0], \
#         reflectivity_file = duet.reflectivity_file['name'], \
#         bandpass_file = duet.bandpass_files['names'][0])
#     >>> test = [0.20659143, 0.37176641]
#     >>> allclose(band_flux, test)
#     True
#  
#     """
# 
# 
#         
# 
#     assert qe_file is not None, 'apply_filters: Need an input QE file'
#     assert reflectivity_file is not None, 'apply_filters: Need an input reflectivity file'
#     assert bandpass_file is not None, 'apply_filters: Need an input bandpass file'
# 
# 
#     # Load filters
#     ref_wave, reflectivity = load_reflectivity(infile = reflectivity_file, **kwargs)
#     qe_wave, qe = load_qe(infile = qe_file, **kwargs)
#     red_wave, red_trans = load_redfilter(infile = bandpass_file, **kwargs)
# 
#     # Apply filters
#     ref_flux = apply_trans(wave, spec, ref_wave, reflectivity)
#     qe_flux = apply_trans(wave, ref_flux, qe_wave, qe)
#     band_flux = apply_trans(wave, qe_flux, red_wave, red_trans)
# 
#     return band_flux


def filter_parameters(duet=None, *args, **kwargs):
    """
    Construct the effective central wavelength and the effective bandpass
    for the filters.
    
    Parameters
    ----------


    Other parameters
    ----------------
    vega : conditional, default False
        Use the Vega spetrum (9.8e3 K blackbody) to compute values. Otherwise computed
        "flat" values if quoting AB mags.
        
    diag : conditional, default False  
        Show the diagnostic info on the parameters.

    Examples
    --------
    >>> band1, band2 = filter_parameters()
    >>> allclose(band1['eff_wave'].value, 202.56878682)
    True
    

    """
    from astroduet.config import Telescope

    if duet is None:
        duet = Telescope()

    from astropy.modeling import models
    from astropy.modeling.blackbody import FNU, FLAM
    from astropy import units as u
    import numpy as np

    vega = kwargs.pop('vega', False)
    diag = kwargs.pop('diag', False)

    wave = np.arange(1000, 10000)*u.AA
    if vega:
        temp = 9.8e3*u.K    
        bb = models.BlackBody1D(temperature=temp)
        flux = bb(wave).to(FLAM, u.spectral_density(wave))
    else:
        flat_model =  np.zeros_like(wave.value)
        flat_model += 1
        flat_model *= FNU
        flux = flat_model.to(FLAM, u.spectral_density(wave))
        


    band1 = duet.apply_filters(wave, flux, band=1)
    band2 = duet.apply_filters(wave, flux, band=2)


    λ_eff1 = ((band1*wave).sum() / (band1.sum())).to(u.nm)
    λ_eff2 = ((band2*wave).sum() / (band2.sum())).to(u.nm)

    dλ = wave[1] - wave[0]
    t1 = band1 / flux
    t2 = band2 / flux

    w1 = (dλ * t1.sum() / t1.max()).to(u.nm)
    w2 = (dλ * t2.sum() / t2.max()).to(u.nm)

    band1 = {'eff_wave': λ_eff1,
             'eff_width': w1}
    band2 = {'eff_wave': λ_eff2,
             'eff_width': w2}

    if diag:
        print('Band1: {0:0.2f} λ_eff, {1:0.2f} W_eff'.format(λ_eff1, w1))
        print('Band2: {0:0.2f} λ_eff, {1:0.2f} W_eff'.format(λ_eff2, w2))

    return band1, band2
















