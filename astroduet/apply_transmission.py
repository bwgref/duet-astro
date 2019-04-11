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