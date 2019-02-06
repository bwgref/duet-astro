def src_rate(diag=False, **kwargs):
    """
    Convert AB magnitude for a source into the number of source counts.
    
    Inputs (defaults):
    ABmag = with Astorpy AB units (22*ur.AB)
    diameter = diameter of mirror (30*ur.cm)
    band = Bandpass (260-330)*ur.nm
    efficiency = Optics efficiency (87%)
    det_qe = Detector quantum efficiency (80%)
    
    
    """
    
    import astropy.units as ur
    import astropy.constants as cr

    import numpy as np
    
    fλ_unit = ur.erg/ur.cm**2/ur.s / ur.Angstrom # Spectral radiances per Hz or per angstrom
    ABmag = kwargs.pop('ABmag', 22*ur.ABmag)
    diameter = kwargs.pop('diameter', 30*ur.cm)
    λ_band = kwargs.pop('band', [260,300]*ur.nm)
        
    elec_per_eV = 1 / (3.6*ur.eV) # per electron for Si
    Area_Tel = np.pi*(diameter.to(ur.cm)*0.5)**2
        
    λ_bandpass = np.abs(λ_band[1] - λ_band[0])
    λ_mid = np.mean(λ_band)
    
    F_λ = ABmag.to(fλ_unit, equivalencies=ur.spectral_density(λ_mid)) # Already converts to flux at this midpoint
    ReceivedPower = (λ_bandpass * F_λ * Area_Tel).to(ur.eV/ ur.s)
    
    ph_energy = (cr.h.cgs * cr.c.cgs / λ_mid.cgs)
    
    NumPh = ReceivedPower / ph_energy.to(ur.eV)
    NumElectrons = ReceivedPower * elec_per_eV
    
    if diag:
        print('Source Computation Integrating over PSF')
        print('Fλ {}'.format(F_λ))
        print('Eff λ {}'.format(λ_mid))
        print('Bandpass: {}'.format(λ_bandpass))
        print('Collecting Area: {}'.format(Area_Tel))
        print('')

    
    
    return NumPh, NumElectrons


def bgd_rate(**kwargs):
    """
    Loads the zodiacal background and normalizes it at 500 nm to a particular
    flux level (low_zodi = 77, med_zodi = 300, high_zodi = 6000), which are taken from
    a paper (to be dropped here later).
    
        
    Optional Inputs (defaults):
    band = Bandpass (180-220)*ur.nm
    diameter=Telescope Diameter (21*u.cm)
    efficiency=Optical throughput (50%)
    pixel_size=Angular size of the pixel (6*ur.arcsec)
    diag = Diagnostics toggle (False)
    
    low_zodi = (True)
    medium_zodi = (False)
    high_zodi = (False)
    
    """
    
    
    import astropy.units as ur
    import astropy.constants as cr
    import numpy as np
    from zodi import load_zodi


    # Set up units here for flux conversion below
    fλ_unit = ur.erg/ur.cm**2/ur.s / ur.Angstrom # Spectral radiances per Hz or per angstrom
    fλ_density_unit = fλ_unit / (ur.arcsec *ur.arcsec)


    diag = kwargs.pop('diag', False)
    
    diameter = kwargs.pop('diameter', 21*ur.cm)

    pixel_size = kwargs.pop('pixel_size', 6*ur.arcsec)
    pixel_area = pixel_size**2

    efficiency = kwargs.pop('efficiecny', 0.5)
    
    

    # Effective Collecting Area of the telescope.
    Area_Tel = efficiency*(diameter.to(ur.cm)*0.5)**2
    
    low_zodi = kwargs.pop('low_zodi', True)
    med_zodi = kwargs.pop('med_zodi', False)
    high_zodi = kwargs.pop('high_zodi', False)
    
    band = kwargs.pop('band', [180,220]*ur.nm)
    bandpass = np.abs(band[1] - band[0])

    effective_wavelength = (np.mean(band)).to(ur.AA)
    ph_energy = (cr.h.cgs * cr.c.cgs / effective_wavelength.cgs).to(ur.eV)

    elec_per_eV = 1 / (3.6*ur.eV) # per electron for Si
    
#     ABmag = 20*ur.ABmag # Just a place holder here
#     F_λ = ABmag.to(fλ_unit, equivalencies=ur.spectral_density(λ_mid)) # Already converts to flux at this midpoint


    if low_zodi:
        zodi_level = 77
    if med_zodi:
        zodi_level = 300
    if high_zodi:
        zodi_level=6000
        
    zodi = load_zodi(scale=zodi_level)

    ctr = 0
    flux_density = 0
    for ind, wv in enumerate(zodi['wavelength']):
        if ( (wv < band[0].to(ur.AA).value ) | (wv > band[1].to(ur.AA).value)):
            continue
        ctr += 1
        flux_density += zodi['flux'][ind]

    # Effective flux density in the band, per arcsecond:
    flux_density /= float(ctr)
    fden = flux_density.to(fλ_density_unit)
    

    ReceivedPower = (bandpass.to(ur.AA) * fden * Area_Tel * pixel_area).to(ur.eV/ ur.s)
    NumPhotons = ReceivedPower / ph_energy # Number of photons
    NumElectrons = ReceivedPower * elec_per_eV
    
    
    if diag:
        print('Background Computation Integrating over Pixel Area')
        print('Fλ total per arcsec2 {}'.format(fden))
        print('Fλ ABmag per arcsec2 {}'.format((F_λ*scale_factor*pixel_area).to(ur.ABmag, equivalencies=ur.spectral_density(λ_mid))))
        print('Bandpass: {}'.format(λ_bandpass))        
        print('Collecting Area: {}'.format(Area_Tel))
        print('Pixel Area: {}'.format(pixel_area))
        print('Photons {}'.format(NumPhotons))
        
    return NumPhotons, NumElectrons

    
    
    
def find_limit(band, **kwargs):

    from astropy import units as ur
    
    import numpy as np
    
#     exposure = kwargs.pop('exposure', 300*ur.s)
    psf_size = kwargs.pop('psf_size', 10*ur.arcsec)
#     snr_limit = kwargs.pop('snr_limit', 5.0)
#     diameter = kwargs.pop('diameter', 21*ur.cm)
#     diag = kwargs.pop('diag', True)
#     
#     aperture_size_pixels = kwargs.pop('aperture_size_pixels', 25.)
#     efficiency = kwargs.pop('efficiency', 0.87)
#     qe = kwargs.pop('det_qe', 0.8)

    pixel_size = psf_size*0.5

    # Returns rates per pixel:
#     nbgd_ph, nbgd_elec = bgd_rate(band=band, diag=False,diameter = diameter,
#         pixel_size = pixel_size, **kwargs)
    nbgd_ph, nbgd_elec = bgd_rate(pixel_size = pixel_size, **kwargs)
    
    return

    # Below is now background electrons in the PSF
    bgd = nbgd_elec * aperture_size_pixels * efficiency * qe * exposure
    
    for dmag in np.arange(1000):
        magtest = (15. + dmag * 0.1) * ur.ABmag
        src_ph, src_elec = src_rate(ABmag=magtest, diag=False,diameter=diameter,band=band, **kwargs)
        
        # Turn this in to electrons per second:
        src = exposure * src_elec * efficiency * qe

        σ = np.sqrt(src + bgd)
        SNR = src / σ

        if SNR < snr_limit:

            if diag:
                print()
                print('Inputs: ')
                print('Exposure {}'.format(exposure))
                print('Optics Diameter: {}'.format(diameter))
                print('PSF Size: {}'.format(psf_size))
                print()
                print('Outputs:')
                print('Source Counts: {}'.format(src))
                print('Background Counts: {}'.format(bgd))
                print('Magnitude limit: {}'.format(magtest))
                print('Signal to noise ratio: {}'.format(SNR))
                print()
            break
    return magtest


def compute_snr(band, ABmag, **kwargs):
    """
    Return the SNR for a specified band at a given *source* ABmag
    
    syntax:
    
    band = [180, 220]*ur.nm
    ABmag = 22*ur.ABmag
    snr = compute_snr(band, ABmag)
    
    
    
    """
    import astropy.units as ur
    import numpy as np
    
    exposure = kwargs.pop('exposure', 300*ur.s)
    psf_size = kwargs.pop('psf_size', 10*ur.arcsec)
    snr_limit = kwargs.pop('snr_limit', 5.0)
    diameter = kwargs.pop('diameter', 30*ur.cm)
    diag = kwargs.pop('diag', False)
    aperture_size_pixels = kwargs.pop('aperture_size_pixels', 25.)
    efficiency = kwargs.pop('efficiency', 0.87)
    qe = kwargs.pop('det_qe', 0.8)

    qe = kwargs.pop('det_qe', 0.8)

    pixel_size = psf_size*0.5

    # Returns rates per pixel:
    nbgd_ph, nbgd_elec = bgd_rate(band=band, diag=False,diameter = diameter, pixel_size = pixel_size, **kwargs)

    # Below is now background electrons in the PSF
    bgd = nbgd_elec * aperture_size_pixels * efficiency * qe * exposure
    
    src_ph, src_elec = src_rate(ABmag=ABmag, diag=False,diameter=diameter,band=band, **kwargs)
    # Turn this in to electrons per second:
    src = exposure * src_elec * efficiency * qe

    σ = np.sqrt(src + bgd)
    SNR = src / σ

    if diag:
    
        print()
        print('Exposure {}:'.format(exposure))
        print('Magnitude test: {}'.format(ABmag))
        print('Optical Efficiency: {}'.format(efficiency))
        print('Source Counts: {}'.format(src))
        print('Background Counts: {}'.format(bgd))
        print('Signal to noise ratio: {}'.format(SNR))

    return SNR



        
    return NumPhotons, NumElectrons