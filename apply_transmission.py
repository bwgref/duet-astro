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
    
    # To do: some checks, wav_s and flux_s must be the same length, wav_t and trans 
    # must be the same length, wav_s and wav_t must be in wavelength units, trans
    # must be between 0 and 1 for all indices.
    
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
