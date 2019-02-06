def src_rate(diag=False, **kwargs):
    """
    Convert AB magnitude for a source into the number of source counts.
    
    Inputs (defaults):
    ABmag = with Astorpy AB units (22*ur.AB)
    diameter = diameter of mirror (30*ur.cm)
    band = Bandpass (260-330)*ur.nm
    efficiency = Optics efficiency (87%)
    det_qe = Detector quantum efficiency (80%)
    
    Returns NumPh, NumElectrons, each of which are ph / cm2 / sec and e- / cm2 / sec
    
    
    """
    
    import astropy.units as ur
    import astropy.constants as cr

    import numpy as np
    
    fλ_unit = ur.erg/ur.cm**2/ur.s / ur.Angstrom # Spectral radiances per Hz or per angstrom
    ABmag = kwargs.pop('ABmag', 22*ur.ABmag)
    λ_band = kwargs.pop('band', [260,300]*ur.nm)
    
    
    diameter = kwargs.pop('diameter', 21.*ur.cm)
    Area_Tel = np.pi * (diameter.to(ur.cm)*0.5)**2

    
    elec_per_eV = 1 / (3.6*ur.eV) # per electron for Si
        
    λ_bandpass = np.abs(λ_band[1] - λ_band[0])
    λ_mid = np.mean(λ_band)
    
    F_λ = ABmag.to(fλ_unit, equivalencies=ur.spectral_density(λ_mid)) # Already converts to flux at this midpoint
    ReceivedPower = (Area_Tel*λ_bandpass * F_λ).to(ur.eV/ ur.s)
    
    ph_energy = (cr.h.cgs * cr.c.cgs / λ_mid.cgs)
    
    NumPh = ReceivedPower / ph_energy.to(ur.eV)
    NumElectrons = ReceivedPower * elec_per_eV
    
    if diag:
        print()
        print('Source Computation Integrating over PSF')
        print('Telescope Diameter: {}'.format(diameter))
        print('Fλ {}'.format(F_λ))
        
        print('Fλ ABmag {}'.format((F_λ).to(ur.ABmag, equivalencies=ur.spectral_density(λ_mid))))
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
    
    
    Returns NumPh, NumElectrons, each of which are ph / cm2 / pixel and e- / cm2 / pxiel
 
    
    """
    
    
    import astropy.units as ur
    import astropy.constants as cr
    import numpy as np
    from zodi import load_zodi


    # Set up units here for flux conversion below
    fλ_unit = ur.erg/ur.cm**2/ur.s / ur.Angstrom # Spectral radiances per Hz or per angstrom
    fλ_density_unit = fλ_unit / (ur.arcsec *ur.arcsec)


    diag = kwargs.pop('diag', False)
    
    pixel_size = kwargs.pop('pixel_size', 6*ur.arcsec)
    pixel_area = pixel_size**2

    diameter = kwargs.pop('diameter', 21.*ur.cm)
    Area_Tel = np.pi*(diameter.to(ur.cm)*0.5)**2

    
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
        print('Telescope diameter: {}'.format(diameter))
        print('Telescope aperture: {}'.format(Area_Tel))
        print('Fλ total per arcsec2 {}'.format(fden))
        print('Fλ ABmag per pixel {}'.format((fden*pixel_area).to(ur.ABmag, equivalencies=ur.spectral_density(effective_wavelength))))
        print('Bandpass: {}'.format(bandpass))        
        print('Collecting Area: {}'.format(Area_Tel))
        print('Pixel Area: {}'.format(pixel_area))
        print('Photons {}'.format(NumPhotons))
        
    return NumPhotons, NumElectrons

    
    
    
def find_limit(band, snr_limit, **kwargs):
    """
    Find the ABmag that results in a 10-sigma detection
    
    Required inputs:
    band = Bandpass (180-220)*ur.nm
    
    Optional Inputs (defaults):
    
    
    
    diameter=Telescope Diameter (21*u.cm)
    efficiency=Optical throughput (50%). Should include ALL photon-path terms.    
    
    pixel_size=Angular size of the pixel (6*ur.arcsec)
    diag = Diagnostics toggle (False)
    psf_size = FWHM of PSF (10. *ur.arcsec)
    ---> pixel size is assumed to be psf_size / 2

    exposure = Exposure time (300*ur.s)
    
    qe = Detector quantum efficiency (80%)

    src_diag = Turn on diagnostics during source computation (False)
    bgd_diag = Turn on diagnostics during background computation (True)
    snr_diag = Report SNR diagnostics (True)

    neff_bgd = Effective aperture background contribution, in pixels (9)

    
    """

    from astropy import units as ur
    
    import numpy as np
    
    
    # PSF size and pixel size
    psf_size = kwargs.pop('psf_size', 10*ur.arcsec)
    pixel_size = psf_size*0.5


    # Effective number of background pixels
    neff_bgd = kwargs.pop('neff_bgd', 9.0)
    

    # Effective Collecting Area of the telescope.
    diameter = kwargs.pop('diameter', 21*ur.cm)
    efficiency = kwargs.pop('efficiency', 0.5)
    
    # Exposure time
    exposure = kwargs.pop('exposure', 300*ur.s)
    
    # Detector Q.E.
    qe = kwargs.pop('qe', 0.8)
    
    EffectiveArea = efficiency*(diameter.to(ur.cm)*0.5)**2

    src_diag = kwargs.pop('src_diag', False)
    bgd_diag = kwargs.pop('bgd_diag', True)
    snr_diag = kwargs.pop('snr_diag', True)


    # Returns rates per pixel:
#     nbgd_ph, nbgd_elec = bgd_rate(band=band, diag=False,diameter = diameter,
#         pixel_size = pixel_size, **kwargs)
    nbgd_ph, nbgd_elec = bgd_rate(band=band, pixel_size = pixel_size,diag=bgd_diag,
        diameter=diameter, **kwargs)
    
    
    
    # nbgd_elec is the number of background electrons per pixel per sec  
    bgd = nbgd_elec * neff_bgd  * exposure * qe * efficiency
    # bgd is now measured electrons during the exposure
    
    for dmag in np.arange(1000):
        magtest = (18. + dmag * 0.1) * ur.ABmag
        src_ph, src_elec = src_rate(ABmag=magtest,
                                    band=band,diameter=diameter,
                                    diag= src_diag)
       # src_elec is the number of electrons / cm2 / sec
        # Turn this in to electrons per second:
        src = src_elec * qe * exposure * efficiency
        sig = np.sqrt(src + bgd)
        SNR = src / sig

        if SNR < snr_limit:
            if snr_diag:
                print()
                print('SNR Computation')
                print('Inputs: ')
                print('Exposure {}'.format(exposure))
                print('Efficiency {}'.format(efficiency))
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
    aperture_size_pixels = kwargs.pop('aperture_size_pixels',9.)
    efficiency = kwargs.pop('efficiency', 0.87)
    qe = kwargs.pop('det_qe', 0.8)

    qe = kwargs.pop('det_qe', 0.8)

    pixel_size = psf_size*0.5

    # Returns rates per pixel:
    nbgd_ph, nbgd_elec = bgd_rate(band=band, diag=False,diameter = diameter,
        pixel_size = pixel_size, **kwargs)

    # Below is now background electrons in the PSF
    bgd = nbgd_elec * aperture_size_pixels * efficiency * qe * exposure
    
    src_ph, src_elec = src_rate(ABmag=ABmag, diag=False,
        diameter=diameter,band=band, **kwargs)
    # Turn this in to electrons per second:
    src = exposure * src_elec * efficiency * qe

    σ = np.sqrt(src + bgd)
    SNR = src / σ

    if diag:
    
        print()
        print('SNR Computation')
        print('Inputs: ')
        print('Exposure {}'.format(exposure))
        print('Efficiency {}'.format(efficiency))
        print('Optics Diameter: {}'.format(diameter))
        print('PSF Size: {}'.format(psf_size))
        print('ABMag: {}'.format(ABmag))
        print()
        print('Outputs:')
        print('Source Counts: {}'.format(src))
        print('Background Counts: {}'.format(bgd))
        print('Signal to noise ratio: {}'.format(SNR))
        print()

    return SNR



        
    return NumPhotons, NumElectrons