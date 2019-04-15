def galex_to_duet(galmags):
    """
    Converts GALEX FUV and NUV magnitude into DUET 1 and DUET 2 magnitude
    Input: array of GALEX magnitudes, either as [[FUV1, FUV2, FUV3, ..., FUVN],[NUV1, NUV2, NUV3,..., NUVN]]
    or as [[FUV1, NUV1],[FUV2, NUV2],...,[FUVN, NUVN]]. 
    Code assumes the first format if len(galmags) = 2, so the second format with two entries will give wrong results!
    Output is in the same format as input but with DUET1 and DUET2 magnitudes instead.
    Conversion is done assuming a linear Fnu, this may break down for sources with a very
    non-linear UV spectrum.
    """

    import astropy.units as u
    import astroduet.config as config
    from astropy.modeling.blackbody import FNU
    import numpy as np
    
    # Setup filters (only interested in effective wavelengths/frequency)
    duet = config.Telescope()
    
    galex_fuv_lef = 151.6 * u.nm
    galex_nuv_lef = 226.7 * u.nm
    
    duet_1_lef = duet.band1['eff_wave']
    duet_2_lef = duet.band2['eff_wave']
    
    galex_fuv_nef = galex_fuv_lef.to(u.Hz, u.spectral())
    galex_nuv_nef = galex_nuv_lef.to(u.Hz, u.spectral())
    
    duet_1_nef = duet_1_lef.to(u.Hz, u.spectral())
    duet_2_nef = duet_2_lef.to(u.Hz, u.spectral())
    
    # Sort input array into FUV and NUV magnitudes
    if len(galmags) == 2:
        fuv_mag = galmags[0]*u.ABmag
        nuv_mag = galmags[1]*u.ABmag
    else:
        fuv_mag = galmags[:,0]*u.ABmag
        nuv_mag = galmags[:,1]*u.ABmag
        
    # Convert GALEX magnitudes to flux densities
    fuv_fnu = fuv_mag.to(FNU, u.spectral_density(galex_fuv_nef))
    nuv_fnu = nuv_mag.to(FNU, u.spectral_density(galex_nuv_nef))
    
    # Extrapolate to DUET bands assuming linear Fnu/nu
    delta_fnu = (nuv_fnu - fuv_fnu)/(galex_nuv_nef - galex_fuv_nef)
    
    d1_fnu = fuv_fnu + delta_fnu*(duet_1_nef - galex_fuv_nef)
    d2_fnu = fuv_fnu + delta_fnu*(duet_2_nef - galex_fuv_nef)
    
    # Convert back to magnitudes
    d1_mag = d1_fnu.to(u.ABmag, u.spectral_density(duet_1_nef))
    d2_mag = d2_fnu.to(u.ABmag, u.spectral_density(duet_2_nef))
    
    # Construct output array
    if len(galmags) == 2:
        duetmags = np.array([d1_mag.value, d2_mag.value])
    else:
        duetmags = np.array([d1_mag.value, d2_mag.value]).transpose()
    
    return duetmags
