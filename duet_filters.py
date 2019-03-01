def make_o2_filter(in_wave,**kwargs):
    """
    Mocks up a filter function to remove the O2 line flux.
    and "rejection" out of band.
    
    Optional inputs (defaults):
    low_wave = lower wavelength of the band (180*ur.nm)
    high_wave = lower wavelength of the band (220*ur.nm)
    rejection = Out of band rejection level (1e-3)
    diag = Give diagnostic info (False)
    
    
    Syntax:
    filter = make_filter(wave)
        
    
    """
    import astropy.units as ur
    import numpy as np

    low_wave = kwargs.pop('low_wave', 2468*ur.AA)
    high_wave = kwargs.pop('high_wave', 2472*ur.AA) 
    rejection = kwargs.pop('rejection', 1e-3)
    diag = kwargs.pop('diag', False)

    
    wave = in_wave.to(ur.Angstrom).value

    low_check = low_wave.to(ur.Angstrom).value
    high_check = high_wave.to(ur.Angstrom).value
    

    red_filter = np.zeros_like(wave)
    red_filter[(wave<low_check) | (wave>high_check)] = 1.0
    red_filter[(wave>=low_check) & (wave<=high_check)] += rejection

    return red_filter


def make_red_filter(in_wave, **kwargs):
    """
    Mocks up a filter function at all of the "wave" points with 1 "in band"
    and "rejection" out of band.
    
    Optional inputs (defaults):
    low_wave = lower wavelength of the band (180*ur.nm)
    high_wave = lower wavelength of the band (220*ur.nm)
    rejection = Out of band rejection level (1e-3)
    diag = Give diagnostic info (False)
    
    
    Syntax:
    filter = make_filter(wave)
        
    
    """
    import astropy.units as ur
    import numpy as np
    blue_filter = kwargs.pop('blue_filter', False)

    
    low_wave = kwargs.pop('low_wave', 180*ur.nm)
    high_wave = kwargs.pop('high_wave', 220*ur.nm) 
    rejection = kwargs.pop('rejection', 1e-3)
    diag = kwargs.pop('diag', False)

    wave = in_wave.to(ur.Angstrom).value

    low_check = low_wave.to(ur.Angstrom).value
    high_check = high_wave.to(ur.Angstrom).value
    

    red_filter = np.zeros_like(wave)
    red_filter[(wave<low_check) | (wave>high_check)] += rejection
    red_filter[(wave>=low_check) & (wave<=high_check)] += 1.0
    
    if blue_filter:
        red_filter[(in_wave < low_wave)] = 1e-6

    
    if diag:
        print('Low wavelength: {}'.format(low_wave))
        print('High wavelength: {}'.format(high_wave))
        print('Rejection level: {}'.format(rejection))
    
    
    return red_filter
    
    
def optimize_filter(low_wave, high_wave, **kwargs):
    """
    Optimizes out-of-band filters based on the input bandpass
    
    ---
    Inputs:
    low_wave = Lower side of bandpass (units consistent with length)

    Option inputs:
    
    qe_band = Which QE file to use (Default is "1")
    target_ratio = Out-of-band to in-band counts (0.5)
    blue_filter = If there's an asymmetric filter, apply this to everything blue-ward
        of the low_wave

    """
    from tdsat_telescope import load_qe, load_reflectivity
    from apply_transmission import apply_trans
    from zodi import load_zodi
    import astropy.units as ur

    # Check if the inputs make sense 
    assert low_wave.unit.is_equivalent(ur.m), "Low-side wavelength does not have unit of length"
    assert high_wave.unit.is_equivalent(ur.m), "High-side wavelength does not have unit of length"

    qe_band = kwargs.pop('qe_band', 1)
    target_ratio = kwargs.pop('target_ratio', 0.5)
    blue_filter = kwargs.pop('blue_filter', False)
    # Load zodiacal background. Note that the out-of-band Zodi dominates over the
    # atmospheric lines (which are present here). Using the lowest Zodi background
    # represents the "worst case".
    zodi = load_zodi()
 
    # Load reflectivity and QE curves:
    ref_wave, reflectivity = load_reflectivity()
    qe_wave, qe = load_qe(band=qe_band)
    
    # Apply these to the Zodi spectrum:
    ref_flux = apply_trans(zodi['wavelength'], zodi['flux'], ref_wave, reflectivity/100.)
    qe_flux = apply_trans(zodi['wavelength'], ref_flux, qe_wave, qe)
 
    # Make a "standard" red filter:
    rejection = 1e-3
    red_filter = make_red_filter(zodi['wavelength'], low_wave = low_wave,
                                 high_wave=high_wave, rejection = rejection,
                                 blue_filter=blue_filter)
    
                                                                 
    band_flux = apply_trans(zodi['wavelength'], qe_flux, zodi['wavelength'], red_filter)
 
    # Get the in-band, out-of-band ratio:
    
    in_band = band_flux[(zodi['wavelength'] > low_wave) &
               (zodi['wavelength']<high_wave)].sum()

    out_of_band = band_flux[((zodi['wavelength'] < low_wave) |
                (zodi['wavelength']>high_wave)) & 
                (zodi['wavelength'] < 1*ur.micron)].sum()

    # Comput ratio:
    ratio = out_of_band / in_band
    target_rejection = (rejection * (target_ratio / ratio)).value
    
    
    
    return target_rejection
