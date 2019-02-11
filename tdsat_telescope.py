def load_telescope_parameters(version, **kwargs):
    """
    Utility script to load the telescope parameters
    
    version = 0: Pre-design version (to compare with Rick's stuff)
    version = 1: 210 mm design
    version = 2: 300 mm design
    version = 3: 350 mm design
    version = 4: 400 mm design

    Syntax:
    diameter, qe, psf_fwhm, efficiency = load_telescope_parameters(version)
    
    """
    
    
    import astropy.units as ur
    from numpy import pi
    
    diag = kwargs.pop('diag', True)
    
    
    if version == 0:
        qe = 0.8 # To be improved later.
        diameter = 30*ur.cm
        psf_fwhm = 10*ur.arcsec
        efficiency = 0.87
    if version == 1:
        qe = 0.8
        efficiency = 0.45
        diameter = 21 * ur.cm
        psf_fwhm = 4 * ur.arcsec
    if version == 2:
        qe = 0.8
        efficiency = 0.67 # Reported from Mike
        diameter = 30 * ur.cm
        psf_fwhm = 9*ur.arcsec
    
    if version == 3:
        qe = 0.8
        diameter = 35*ur.cm
        efficiency = 0.70 # Assumed
        psf_fwhm = 18*ur.arcsec

    if version == 4:
        qe = 0.8
        diameter = 40*ur.cm
        efficiency = 0.75 # Assumed
        psf_fwhm = 23*ur.arcsec

    if diag:
        print('Telescope Configuration {}'.format(version))
        print('Entrance Pupil diameter {}'.format(diameter))
        print('Optical Effifiency {}'.format(efficiency))
        print('PSF FWHM {}'.format(psf_fwhm))
        print('Effective Aperture {}'.format(diameter*(efficiency)**0.5))
        print('Effective Area {}'.format( efficiency * pi * (0.5*diameter)**2))
              
    return diameter, qe, psf_fwhm, efficiency