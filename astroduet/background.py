from numpy import allclose

def background_pixel_rate(duet, **kwargs):    
    """
    Determine the background rate per pixel.
    
    Parameters
    ----------

    duet: astroduet.config.Telescope instance
        The description of the Telescope

    Other parameters
    ----------------        
    diag : conditional, default False  
        Show the diagnostic info on the parameters.
    
    low_zodi: coniditonal , default is True 
        Use the low zodiacal background rate

    med_zodi: coniditonal , default is False 
        Use the medium zodiacal background rate. Overrides low_zodi.

    high_zodi: coniditonal , default is False 
        Use the medium zodiacal background rate. Overrideslow_zodi.
        
    Returns
    -------
    pixel_rate : 1-D float
        ``[band1_rate, band2_rate]``
        The count rate per pixel after the filtering has been applied in each band.
        
    Examples
    --------
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> [bgd1, bgd2] = background_pixel_rate(duet, high_zodi=True)
    >>> allclose(bgd1.value, 0.190, atol=0.001)
    True
    

    """
    
    from astroduet.config import Telescope
    from astroduet.filters import apply_filters
    assert isinstance(duet, Telescope), "First parameter needs to be a astroduet.config.Telescope class"
    import astropy.units as u
    from astroduet.zodi import load_zodi

    diag = kwargs.pop('diag', False)
    low_zodi = kwargs.pop('low_zodi', True)
    med_zodi = kwargs.pop('med_zodi', False)
    high_zodi = kwargs.pop('high_zodi', False)
    
    if low_zodi:
        zodi_level = 77
    if med_zodi:
        zodi_level = 165
    if high_zodi:
        zodi_level=900
    zodi = load_zodi(scale=zodi_level)
    wave = zodi['wavelength'] 
    flux = zodi['flux']
    
    band1_flux = apply_filters(zodi['wavelength'], zodi['flux'], band=1, **kwargs)
    band2_flux = apply_filters(zodi['wavelength'], zodi['flux'], band=2, **kwargs)
    
#     # Assume bins are the same size:
    de = wave[1] - wave[0]    
#     # Convert to more convenient units:
    ph_flux1 = ((de*band1_flux).cgs).to(1 / ((u.cm**2 * u.arcsec**2 * u.s)))
    ph_flux2 = ((de*band2_flux).cgs).to(1 / ((u.cm**2 * u.arcsec**2 * u.s)))

    # Compute fluence
    fluence1 = ph_flux1.sum()
    fluence2 = ph_flux2.sum()

    pixel_area = duet.pixel**2
    eff_area =duet.eff_area
    trans_eff = duet.trans_eff

    # Apply telescope values    
    bgd_rate1 = eff_area * pixel_area * fluence1 * trans_eff
    bgd_rate2 = eff_area * pixel_area * fluence2 * trans_eff

    if diag:
        print('Background Computation Integrating over Pixel Area')
        print('Telescope diameter: {}'.format(duet.EPD))
        print('Collecting Area: {}'.format(duet.eff_area))
        print('Transmission Efficiency: {}'.format(duet.trans_eff))
        
        print()
        print()
        print('Pixel Size: {}'.format(duet.pixel))
        print('Pixel Area: {}'.format(pixel_area))
        print()
        print('Zodi Level: {}'.format(zodi_level))
        print('Band1 Rate: {}'.format(bgd_rate1))
        print('Band2 Rate: {}'.format(bgd_rate2))
        
        
    return [bgd_rate1, bgd_rate2]
