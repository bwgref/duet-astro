def src_rate(diag=False, **kwargs):
    """
    Convert AB magnitude for a source into the number of source counts.
    
    Inputs (defaults):
    ABmag = with Astorpy AB units (22*ur.AB)
    diameter = diameter of mirror (30*ur.cm)
    band = Bandpass (260-330)*ur.nm
    efficiency = Optics efficiency (87%)
    det_qe = Detector quantum efficiency (80%)
    
    Returns NumPh, NumElectrons, each of which are ph / sec and e- / sec
    
    Note that you have to include optical efficiency and quantum efficiency
    *after* this step.
    
    Second Note: Eventually this needs a more sophisticated.
    
    
    """
    
    import astropy.units as ur
    import astropy.constants as cr

    import numpy as np
    
    fλ_unit = ur.erg/ur.cm**2/ur.s / ur.Angstrom # Spectral radiances per Hz or per angstrom
    ABmag = kwargs.pop('ABmag', 22*ur.ABmag)
    λ_band = kwargs.pop('band', [180,220]*ur.nm)
    
    
    diameter = kwargs.pop('diameter', 21.*ur.cm)
    Area_Tel = np.pi * (diameter.to(ur.cm)*0.5)**2

    
#    elec_per_eV = 1 / (3.6*ur.eV) # per electron for Si
    elec_per_eV = 1 / (1*ur.eV) # per electron for Si
 
 
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
        print('ReceivedPower: {}'.format(ReceivedPower))
        print('Photons per second: ', NumPh)
        print('Photons per second per cm2: ', NumPh / Area_Tel)
        
        print('Fλ ABmag {}'.format((F_λ).to(ur.ABmag, equivalencies=ur.spectral_density(λ_mid))))
        print('Eff λ {}'.format(λ_mid))
        print('Bandpass: {}'.format(λ_bandpass))
        print('Collecting Area: {}'.format(Area_Tel))
        print('')

    
    
    return NumPh, NumElectrons





def bgd_electronics(exposure, **kwargs):
    """
    Place holder to account for dark currents, etc
    
    Required inputs:
    exposure = Integration time of the frame (i.e. 300*ur.sc)
    
    Optional inputs:
    
    dark_current = Dark current in electrons per pixel per second (0.065)
    read noise = read noise in eelcetrons per frame (33)

    Returns: Number of background electrons 

    Syntax:
    
    nelec = bgd_electronics(300*ur.s)

    """
    import astropy.units as ur
    dark_current = kwargs.pop('dark_current', 0.065 / (1*ur.s)) # electrons per pixel
    read_noise = kwargs.pop('read_noise', 33.)
    diag = kwargs.pop('diag', False)
    
    # Just dark current for now...
    nelec = dark_current *exposure + read_noise
    
    if diag:
        print('Dark current (electrons / sec) {}'.format(dark_current))
        print('Exposure {}'.format(exposure))
        print('Dark electrons (electrons) {}'.format(dark_current*exposure))
        print('Read noise (electrons / pixel / frame) {}'.format(read_noise))
        print('Total electronis bgd (electrons) {}'.format(nelec))
    

    

    return nelec    



def bgd_sky_rate(**kwargs):
    """
    Loads the zodiacal background and normalizes it at 500 nm to a particular
    flux level (low_zodi = 77, med_zodi = 300, high_zodi = 6000), which are taken from
    a paper (to be dropped here later).
    
        
    Optional Inputs (defaults):
    band = Bandpass (180-220)*ur.nm
    diameter=Telescope Diameter (21*u.cm)
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
        zodi_level = 165
    if high_zodi:
        zodi_level=900
        
    zodi = load_zodi(scale=zodi_level)

    ctr = 0
    flux_density = 0
    for ind, wv in enumerate(zodi['wavelength']):
        if ( (wv < band[0].to(ur.AA) ) | (wv > band[1].to(ur.AA))):
            continue
        ctr += 1
        flux_density += zodi['flux'][ind]

    # Effective flux density in the band, per arcsecond:
    flux_density /= float(ctr)
    fden = flux_density.to(fλ_density_unit)
    

    ReceivedPower = (bandpass.to(ur.AA) * fden * Area_Tel * pixel_area).to(ur.eV/ ur.s)
    NumPhotons = ReceivedPower / ph_energy # Number of photons


    NumPhotons = NumPhotons 
    
    ElectronsPerPhoton = (ph_energy.to(ur.eV)) * elec_per_eV 

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





def bgd_sky_qe_rate(**kwargs):
    """
    Loads the zodiacal background and normalizes it at 500 nm to a particular
    flux level (low_zodi = 77, med_zodi = 300, high_zodi = 6000).
    
    See the docstring for load_zodi in zodi.py 
        
    Optional Inputs (defaults):
    band = Bandpass (180-220)*ur.nm
    diameter=Telescope Diameter (21*u.cm)
    pixel_size=Angular size of the pixel (6*ur.arcsec)
    rejection = Out of band rejection (1e-3)
    diag = Diagnostics toggle (False)
    
    low_zodi = (True)
    medium_zodi = (False)
    high_zodi = (False)
    
    qe_band = Whivch QE curve to us (1 --> 180-220 nm, 2-->260-300 nm)
    blue_filter = Apply blue_side filter (False)
    
    
    Returns bgd_rate which is ph / s / pixel 
    
    """
    
    
    import astropy.units as ur
    import astropy.constants as cr
    import numpy as np
    from zodi import load_zodi, wavelength_to_energy
    from apply_transmission import apply_trans
    from tdsat_telescope import load_qe, load_reflectivity, load_redfilter, apply_filters
    from duet_filters import make_red_filter, optimize_filter


    # Set up units here for flux conversion below
#    fλ_unit = ur.erg/ur.cm**2/ur.s / ur.Angstrom # Spectral radiances per Hz or per angstrom
#    fλ_density_unit = fλ_unit / (ur.arcsec *ur.arcsec)

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
    
    qe_band = kwargs.pop('qe_band', 1)
    blue_filter = kwargs.pop('blue_filter', False)
    
    filter_target = kwargs.pop('filter_target', 0.5)

    real_red = kwargs.pop('real_red', False)

#    effective_wavelength = (np.mean(band)).to(ur.AA)
#    ph_energy = (cr.h.cgs * cr.c.cgs / effective_wavelength.cgs).to(ur.eV)

    # Specified from Kristin. The highest Zodi that we hit is actually only 900 
    # (down from 6000 in previous iterations) sine any closer to the Sun we violate
    # our Sun-angle constraints.
    
    if low_zodi:
        zodi_level = 77
    if med_zodi:
        zodi_level = 165
    if high_zodi:
        zodi_level=900
        
    zodi = load_zodi(scale=zodi_level)

    wave = zodi['wavelength'] 
    flux = zodi['flux'] 
    
    
    
    
    
    if real_red:
        band_flux = apply_filters(zodi['wavelength'], zodi['flux'], band=qe_band, diag=diag, **kwargs)
    else:

    # Make the red filter
        low_wave = band[0]
        high_wave = band[1]
        rejection = optimize_filter(low_wave, high_wave, target_ratio=filter_target, 
            blue_filter=blue_filter)
            
        
    
        red_trans = make_red_filter(wave, rejection=rejection, high_wave = high_wave,
                low_wave = low_wave, blue_filter = blue_filter)
        red_wave = wave
            # Load reflectivity and QE curves:
        ref_wave, reflectivity = load_reflectivity()
        qe_wave, qe = load_qe(band=qe_band)

        
        # Apply reflectivity and QE to the Zodi spectrum:
        ref_flux = apply_trans(zodi['wavelength'], zodi['flux'], ref_wave, reflectivity/100.)
        qe_flux = apply_trans(zodi['wavelength'], ref_flux, qe_wave, qe)
    
        # Apply red filter
        band_flux = apply_trans(wave, qe_flux, red_wave, red_trans)

    # Assume bins are the same size:
    de = wave[1] - wave[0]    
    # Convert to more convenient units:
    ph_flux = ((de*band_flux).cgs).to(1 / ((ur.cm**2 * ur.arcsec**2 * ur.s)))
    fluence = ph_flux.sum()

    BgdRatePerPix = pixel_area * fluence * Area_Tel
    

    if diag:
        print('Background Computation Integrating over Pixel Area')
        print('Telescope diameter: {}'.format(diameter))
        print('Collecting Area: {}'.format(Area_Tel))
        print('Band: {}'.format(band))
        print('Bandpass: {}'.format(bandpass))
        print()
#        print('Out-of-band rejection: {}'.format(rejection))
#        print('Apply blue filter? {}'.format(blue_filter))
        print()
        print('Pixel Area: {}'.format(pixel_area))
        print()
        print('Background fluence per arcsec2 {}'.format(fluence))
        print('Rate {}'.format(BgdRatePerPix))
        
    return BgdRatePerPix

    
def outofband_bgd_sky_rate(**kwargs):
    """
    Loads the zodiacal background and normalizes it at 500 nm to a particular
    flux level (low_zodi = 77, med_zodi = 300, high_zodi = 6000), which are taken from
    a paper (to be dropped here later).
    Calculates out-of-band contribution to the background (up to a maximum wavelength, default 900 nm).
    Out-of-band rejection efficiency is folded in later, in the snr calculation.
        
    Optional Inputs (defaults):
    band = Bandpass (180-220)*ur.nm
    diameter=Telescope Diameter (21*ur.cm)
    pixel_size=Angular size of the pixel (6*ur.arcsec)
    max_wav = cutoff wavelength of detector (900*ur.nm)
    diag = Diagnostics toggle (False)
    
    low_zodi = (True)
    medium_zodi = (False)
    high_zodi = (False)
    
    
    Returns NumPh, NumElectrons, each of which are ph / cm2 / pixel and e- / cm2 / pixel
 
    
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
    max_wav = kwargs.pop('max_wav', 900.*ur.nm)
    
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
        zodi_level = 165
    if high_zodi:
        zodi_level=900
        
    zodi = load_zodi(scale=zodi_level)

    ctr = 0
    flux_density = 0
    for ind, wv in enumerate(zodi['wavelength']):
        if ( (wv >= band[0].to(ur.AA).value ) & (wv <= band[1].to(ur.AA).value) | (wv >= max_wav.to(ur.AA).value)):
            continue
        ctr += 1
        flux_density += zodi['flux'][ind]

    # Effective flux density in the band, per arcsecond:
    flux_density /= float(ctr)
    fden = flux_density.to(fλ_density_unit)
    
    outofbandpass = np.abs(max_wav - zodi['wavelength'][0]*ur.AA)-bandpass

    ReceivedPower = (outofbandpass.to(ur.AA) * fden * Area_Tel * pixel_area).to(ur.eV/ ur.s)
    NumPhotons = ReceivedPower / ph_energy # Number of photons


    NumPhotons = NumPhotons 
    
    ElectronsPerPhoton = (ph_energy.to(ur.eV)) * elec_per_eV 

    NumElectrons = ReceivedPower * elec_per_eV 
    
    if diag:
        print('')
        print('Out of band Background Computation Integrating over Pixel Area')
        print('Telescope diameter: {}'.format(diameter))
        print('Telescope aperture: {}'.format(Area_Tel))
        print('Fλ total per arcsec2 {}'.format(fden))
        print('Fλ ABmag per pixel {}'.format((fden*pixel_area).to(ur.ABmag, equivalencies=ur.spectral_density(effective_wavelength))))
        print('Bandpass: {}'.format(bandpass))
        print('Detector cutoff wavelength: {}'.format(max_wav))        
        print('Collecting Area: {}'.format(Area_Tel))
        print('Pixel Area: {}'.format(pixel_area))
        print('Photons {}'.format(NumPhotons))
        
    return NumPhotons, NumElectrons



    
def compute_snr(band, ABmag, **kwargs):
    """
    Return the SNR for a specified band at a given *source* ABmag
    
    syntax:
    
    import astropy.units as ur
    band = [180, 220]*ur.nm
    ABmag = 22*ur.ABmag
    snr = compute_snr(band, ABmag)
    
    ---
    Optional inputs:
    diag = SNR calculation diagnositcs (False)
    bgd_diag = Background estimation diagnostics (False)
    bgd_elec_diag = Background electronics diagnostics (False)
    src_diag = Source counts diagnostics (False)
    
    psf_size = PSF FWHM (10 *u.arcsec)
    pixel_size = Pixel size (5*ur.arcsec)
    exposure = Frame exposure time (300 *ur.s)
    diameter = Entrance pupil diamtere (30 * ur.cm)
    frames = Number of frames to difference (1, i.e. perfect background subtraction)
    
    efficiency = Total optical efficiency. Note that this should currently include
        reflectivity, transmission through the correctors, and the geometric
        obscuration of the beam by the focal plane. The latter is from load_telescope.
        
    """
    import astropy.units as ur
    import numpy as np
    from tdsat_neff import get_neff


    diag = kwargs.pop('diag', False)
    bgd_diag = kwargs.pop('bgd_diag', False)
    bgd_elec_diag = kwargs.pop('bgd_elec_diag', False)
    src_diag = kwargs.pop('src_diag', False)


    # Telescope and observation params
    psf_size = kwargs.pop('psf_size', 10*ur.arcsec)
    pixel_size = kwargs.pop('pixel_size', 5*ur.arcsec)
    qe = kwargs.pop('det_qe', 0.8)

    exposure = kwargs.pop('exposure', 300*ur.s)
    diameter = kwargs.pop('diameter', 30*ur.cm)

    frames = kwargs.pop('frames', 1)
    efficiency = kwargs.pop('efficiency', 0.8)

    # To be depricated
    outofband_qe = kwargs.pop('outofband_qe', 0.001)
    neff_bgd = kwargs.pop('neff_bgd',5.6)

    neff_bgd =  get_neff(psf_size, pixel_size)
    
    # Returns rates per pixel due to sky backgrounds
    #nbgd_ph, nbgd_elec = bgd_sky_rate(band=band, diag=bgd_diag,diameter = diameter,
    #    pixel_size = pixel_size, **kwargs)

    # Returns rates per pixel due to out-of-band sky backgrounds
    #noobbgd_ph, noobbgd_elec = outofband_bgd_sky_rate(band=band, diag=bgd_diag,diameter = diameter,
    #    pixel_size = pixel_size, **kwargs)

    # Use the updated sky background calculator with qe already folded in
    nbgd_ph = bgd_sky_qe_rate(band=band, diag=bgd_diag, diameter = diameter,
        pixel_size = pixel_size, rejection = outofband_qe, **kwargs)
    
    # Get background due to electronics:
    bgd_elec = bgd_electronics(exposure, diag=bgd_elec_diag, **kwargs)
    
    # Below is now background PHOTONS in the PSF and the electronics background
    # rates (in electrons).
    bgd = nbgd_ph * efficiency * exposure + bgd_elec 
    
    src_ph, src_elec = src_rate(ABmag=ABmag, diag=src_diag,
        diameter=diameter,band=band, **kwargs)
    # Turn this in to electrons per second:
    src = exposure * src_ph * efficiency * qe

    # bgd has the dark current and the zodiacal background in there.
    # For now, assume that you're looking at the difference of two frames, hence the
    # factor of two below.
    σ = np.sqrt(src + frames * neff_bgd * bgd)
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
        print('Source rate (ph / s): {}'.format(src_ph * efficiency * qe))
        print('Source Photons: {}'.format(src_ph*efficiency*qe*exposure))
        print('Source Electrons: {}'.format(src_elec*efficiency*qe*exposure))

        print()
        print('In-band sky background photons: {}'.format(neff_bgd * nbgd_ph  * efficiency * qe * exposure))
        print('Out-of-band sky background photons: {}'.format(neff_bgd*noobbgd_ph * efficiency * outofband_qe * exposure))
        print('Electronics Background: {}'.format(neff_bgd*bgd_elec))
        print('Total Background Counts: {}'.format(bgd*neff_bgd))

        print()
        print('Signal to noise ratio: {}'.format(SNR))
        print()

    return SNR


    
def find_limit(band, snr_limit, **kwargs):
    """
    Find the ABmag that results in a 10-sigma detection.
        
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
    neff_bgd = kwargs.pop('neff_bgd', 5.6)
    
    bgd_diag = kwargs.pop('bgd_diag', False)
    src_diag = kwargs.pop('src_diag', False)
    bgd_elec_diag = kwargs.pop('bgd_elec_diag', False)
    snr_diag = kwargs.pop('snr_diag', False)
    outofband_qe = kwargs.pop('outofband_qe', 0.001)

    frames = kwargs.pop('frames', 2)

    # Effective Collecting Area of the telescope default settings
    diameter = kwargs.pop('diameter', 21*ur.cm)
    efficiency = kwargs.pop('efficiency', 0.5)
    
    # Exposure time
    exposure = kwargs.pop('exposure', 300*ur.s)
    
    # Detector Q.E.
    qe = kwargs.pop('qe', 0.8)
    
    EffectiveArea = efficiency*(diameter.to(ur.cm)*0.5)**2


    # Only compute backgrounds once, instead of every loop:

    # Returns rates per pixel due to sky backgrounds (just zodiacal for now)
    nbgd_ph, nbgd_elec = bgd_sky_rate(band=band, pixel_size = pixel_size,diag=bgd_diag,
        diameter=diameter, **kwargs)
    
    
    # Returns rates per pixel due to out-of-band sky backgrounds
    noobbgd_ph, noobbgd_elec = outofband_bgd_sky_rate(band=band, diag=bgd_diag,diameter = diameter,
        pixel_size = pixel_size, **kwargs)

    # Get background due to electronics:
    bgd_elec = bgd_electronics(exposure, diag=bgd_elec_diag)
 
    # Get total background rate per pixel (efficiency and q.e. don't operatre
    # on electronic noise):
    bgd = nbgd_ph  * efficiency * qe * exposure + noobbgd_ph * efficiency * outofband_qe * exposure + bgd_elec

    # Dumb loop here. Start bright and keep getting fainter until you hit 10-sigma   

    for dmag in np.arange(1000):
        magtest = (18. + dmag * 0.1) * ur.ABmag

        src_ph, src_elec = src_rate(ABmag=magtest, diag=src_diag,
            diameter=diameter,band=band, **kwargs)

       # src_elec is the number of electrons / sec
        src = src_ph * qe * exposure * efficiency
        sig = np.sqrt(src + frames*neff_bgd*bgd)
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


def calc_exposure(k, src_rate, bgd_rate, read_noise, neff):
    """
    Compute the time to get to a given significance (k) given the source rate,
    the background rate, the read noise, and the number
    of effective background pixels
    
    -----
    
    time = calc_exposure(k, src_rate, bgd_rate, read_noise, neff)
    
    """
    denom = 2 * src_rate**2

    nom1 = (k**2) * (src_rate + neff*bgd_rate)
    
    nom2 = ( k**4 *(src_rate + neff*bgd_rate)**2 +
                    4 * k**2 * src_rate**2 * neff * read_noise**2)**(0.5)
    exposure = (nom1 + nom2)/ denom
    return exposure

def calc_snr(texp, src_rate, bgd_rate, read_noise, neff):
    """
    Compute S/N given the exposure time, source rate,
    the background rate, the read noise, and the number
    of effective background pixels and the number of stacked exposures

    -----
    
    snr = calc_snr(texp, src_rate, bgd_rate, read_noise, neff, nexp)
    
    """
    denom = (src_rate*texp + neff * (bgd_rate*texp + read_noise**2))**0.5

    nom = src_rate * texp
    
    snr = nom / denom
    return snr



