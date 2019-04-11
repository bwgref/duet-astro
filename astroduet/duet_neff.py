def load_neff():
    """
    Load number of effective background pixels in the PSF from
    file provided by Rick Cook.
    
    ----
    Returns
    
    oversample is the ration of the PSF to the pixel size.
    neff is is the resulting value.
    
    """
    import os
    from numpy import genfromtxt
    ref_file = os.path.join('input_data', 'neff_data.dat')
    header=True
    neff = {}
    oversample, neff = genfromtxt(ref_file, unpack=True, skip_header=True)
    return oversample, neff
    
    
def get_neff(psf_size, pixel_size):
    """
    Determine the number of effective background pixels based on the PSF size and the
    pixel size. Assume these are given with astropy units:
    
    ---
    
    """
    from numpy import interp
    over, neff = load_neff()
    data_oversample = (psf_size / pixel_size).value
        
    neff = interp(data_oversample, over, neff)
    return neff
    

        