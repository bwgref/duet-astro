import glob
import os
import numpy as np


curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'data')


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

    zodi_airglow : boolean
        Use the old version of the airglow lines (Default is False)
        
    airglow_level : string
        Which of the new airglow levels do we want to use.
        Options are: False, 'Low', 'Average', and 'High'
    


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
    >>> allclose(bgd1.value, 0.05221417, atol=0.001)
    True


    """

    from astroduet.config import Telescope
    assert isinstance(duet, Telescope), "First parameter needs to be a astroduet.config.Telescope class"
    import astropy.units as u
    from astroduet.zodi import load_zodi

    diag = kwargs.pop('diag', False)
    low_zodi = kwargs.pop('low_zodi', True)
    med_zodi = kwargs.pop('med_zodi', False)
    high_zodi = kwargs.pop('high_zodi', False)
    
    airglow_level = kwargs.pop('airglow_level', 'Average')
    zodi_airglow = kwargs.pop('zodi_airglow', False)

    if low_zodi:
        zodi_level = 77
    if med_zodi:
        zodi_level = 165
    if high_zodi:
        zodi_level=900
    zodi = load_zodi(scale=zodi_level, airglow=zodi_airglow)
    wave = zodi['wavelength']
    flux = zodi['flux']

    if airglow_level is not False:
        airglow_flux = airglow_lines(wave, airglow_level=airglow_level)
        flux += airglow_flux

    band1_flux = duet.apply_filters(zodi['wavelength'], flux, band=1, **kwargs)
    band2_flux = duet.apply_filters(zodi['wavelength'], flux, band=2, **kwargs)

#     # Assume bins are the same size:
    de = wave[1] - wave[0]
#     # Convert to more convenient units:
    ph_flux1 = (de*band1_flux).to(u.ph / ((u.cm**2 * u.arcsec**2 * u.s)))
    ph_flux2 = (de*band2_flux).to(u.ph / ((u.cm**2 * u.arcsec**2 * u.s)))

    # Compute fluence
    fluence1 = ph_flux1.sum()
    fluence2 = ph_flux2.sum()

    pixel_area = duet.pixel**2

    bgd_rate1 = duet.fluence_to_rate(pixel_area * fluence1)
    bgd_rate2 = duet.fluence_to_rate(pixel_area * fluence2)

    if diag:
        print('-----')
        print('Background Computation Integrating over Pixel Area')
        print('Telescope diameter: {}'.format(duet.EPD))
        print('Collecting Area: {}'.format(duet.eff_area))
#        print('Transmission Efficiency: {}'.format(duet.trans_eff))

        print()
        print()
        print('Pixel Size: {}'.format(duet.pixel))
        print('Pixel Area: {}'.format(pixel_area))
        print()
        print('Zodi Level: {}'.format(zodi_level))
        print('Band1 Rate: {}'.format(bgd_rate1))
        print('Band2 Rate: {}'.format(bgd_rate2))
        print('-----')



    return [bgd_rate1, bgd_rate2]

def airglow_lines(wave, airglow_level='Average'):
    """
    Returns the airglow contribution

    Parameters
    ----------

    wave : array
        Should have Astropy units.

    level: string
        'Low', 'Average' (default), or 'High'


    Returns
    -------
    airglow_spec : array
        The surface brightenss in units of ph / cm2 / sec / sr / AA

    """
    from astropy import units as u

    levels = ['High', 'Average', 'Low', 'COS']
    assert airglow_level in levels
    flux_ind = levels.index(airglow_level) + 1 # +1 to offset from wavelength column
    
    step = (wave[1] - wave[0]).to(u.AA).value
    fl_unit = u.ph / (u.s * u.cm**2 * u.sr * u.AA)
    
    airglow_spec = np.zeros_like(wave.value)*fl_unit
    
    airglow = np.genfromtxt(datadir+'/airglow_lines.txt', skip_header=2)    
    for row in airglow:
        idx = (np.abs((wave - row[0]*u.AA).value)).argmin()
        airglow_spec[idx] += row[flux_ind] * fl_unit / step
    
    return airglow_spec
