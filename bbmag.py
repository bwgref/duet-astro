

def bb_abmag(diag=False, val=False, **kwargs):
    """
    Take a blackbody with a certain temperature and convert to AB magnitudes in two bands.
    Scaled to u-band or Swift UVW2 magnitude.
    
    Inputs (defaults):
    umag = apparent u-band AB magnitude (22*ur.ABmag)
    swiftmag = apparent Swift UVW2 magnitude (22*ur.ABmag)
    ref = band to use for reference magnitude; options are 'u', 'swift' ('swift')
    bandone = Bandpass 1st filter (180-220)*ur.nm
    bandtwo = Bandpass 2nd filter (260-300)*ur.nm
    bbtemp = Blackbody temperature (20000*ur.K)
    dist = Distance (10*ur.pc)
    val = Return values without unit if True (False)
    bolflux = Bolometric flux; if not 1, refmag and distance are ignored (1*ur.erg/ur.cm**2/ur.s)
    
    diag (False)
        
    Returns ABmag1, ABmag2
       
    Also still to do: Add background from galaxies.
       
    """
    
    import astropy.units as ur
    import astropy.constants as cr
    from astropy.modeling import models
    from astropy.modeling.blackbody import FLAM
    import numpy as np
    
    bbtemp = kwargs.pop('bbtemp', 20000.*ur.K)
    bandone = kwargs.pop('bandone', [180,220]*ur.nm)
    bandtwo = kwargs.pop('bandtwo', [260,300]*ur.nm)
    umag = kwargs.pop('umag', 22*ur.ABmag)
    swiftmag = kwargs.pop('swiftmag', 22*ur.ABmag)
    dist = kwargs.pop('dist', 10*ur.pc)
    ref = kwargs.pop('ref', 'swift')
    dist0 = 10*ur.pc
    bolflux = kwargs.pop('bolflux', 1.*ur.erg/(ur.cm**2 * ur.s))
    
    bandu = [340,380]*ur.nm # For comparison purposes
    bandsw = [172.53,233.57]*ur.nm # Swift UVW2 effective band (lambda_eff +/- 0.5 width_eff)

    wav = np.arange(1000,9000) * ur.AA # Wavelength scale in 1 Angstrom steps
    bb = models.BlackBody1D(temperature=bbtemp,bolometric_flux=bolflux) # Load the blackbody model
    flux = bb(wav).to(FLAM, ur.spectral_density(wav))

    # Calculate mean flux density in each band:
    fluxden_one = np.mean(flux[(wav >= bandone[0].to(ur.AA)) & (wav <= bandone[1].to(ur.AA))])
    fluxden_two = np.mean(flux[(wav >= bandtwo[0].to(ur.AA)) & (wav <= bandtwo[1].to(ur.AA))])
    fluxden_u = np.mean(flux[(wav >= bandu[0].to(ur.AA)) & (wav <= bandu[1].to(ur.AA))])
    fluxden_sw = np.mean(flux[(wav >= bandsw[0].to(ur.AA)) & (wav <= bandsw[1].to(ur.AA))])

    # Convert to AB magnitudes:
    magone = fluxden_one.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandone)))
    magtwo = fluxden_two.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandtwo)))
    magu = fluxden_u.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandu)))
    magsw = fluxden_sw.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandsw)))
    
    if (ref == 'u'):
    # Offset from comparison u-band magnitude:
        magoff = umag - magu
    elif (ref =='swift'):
    # Offset from comparison swift UVW2-band magnitude:
        magoff = swiftmag - magsw

    # Distance modulus
    distmod = (5*np.log10(dist/dist0)).value*ur.mag

    # Apply offsets
    magone_final = magone + magoff + distmod
    magtwo_final = magtwo + magoff + distmod

    if (bolflux == 1.*ur.erg/(ur.cm**2 * ur.s)):
        magone_final = magone + magoff + distmod
        magtwo_final = magtwo + magoff + distmod
    else:
        magone_final = magone
        magtwo_final = magtwo
 
    if diag:
        print()
        print('Compute ABmags in TD bands for blackbody')
        print('Blackbody temperature: {}'.format(bbtemp))
        print('Reference UVW2-band magnitude: {}'.format(swiftmag))
        print('Band one: {}'.format(bandone))
        print('Band two: {}'.format(bandtwo))
        print('Distance: {}'.format(dist))
        
        print('Flux density band one: {}'.format(fluxden_one))
        print('Flux density band two: {}'.format(fluxden_two))
        print('Flux density Swift: {}'.format(fluxden_sw))
        print('Distance modulus: {}'.format(distmod))
        print('Raw ABmag band one: {}'.format(magone))
        print('Raw ABmag band two: {}'.format(magtwo))
        print('Raw ABmag Swift: {}'.format(magsw))
        print('Offset from Swift band: {}'.format(magoff))
        print('ABmag band one: {}'.format(magone_final))
        print('ABmag band two: {}'.format(magtwo_final))
        print('')    
    
    if val:
        return magone_final.value, magtwo_final.value
    else:
        return magone_final, magtwo_final

def sigerr(snr):
    """
    Takes S/N ratio (from compute_snr) and returns it as a magnitude error
    
    Input:
    snr = signal-to-noise
    """
    
    import numpy as np
    import astropy.units as ur
    
    sigma = 2.5*np.log10(1.+1./snr.value)*ur.ABmag
    
    return sigma
    
def bbfunc(x,*par):
    """
    Helper function for gettempbb. Initialize a blackbody model without values.
    """
    import astropy.units as ur
    from astropy.modeling import models
    from astropy.modeling.blackbody import FLAM   
    
    temp,norm = par
    mod = models.BlackBody1D(temperature=temp*ur.K,bolometric_flux = norm*ur.erg/(ur.cm**2 * ur.s))
    return mod(x*ur.nm).to(FLAM, equivalencies=ur.spectral_density(x*ur.nm)).value
    
def gettempbb(diag=False, val=False, **kwargs):
    """
    Take AB magnitudes in two bands (with errorbars) and fit a blackbody to retrieve the temperature.
    Assumes no reddening.
    
    Inputs (defaults):
    bandone = Bandpass 1st filter (180-220)*ur.nm
    bandtwo = Bandpass 2nd filter (260-300)*ur.nm
    magone = AB magnitude in band one (22*ur.ABmag) 
    magtwo = AB magnitude in band two (22*ur.ABmag)
    magone_err = error on the band one magnitude (0.1*ur.ABmag)  
    magtwo_err = error on the band two magnitude (0.1*ur.ABmag)
    bbtemp_init = Initial BBtemperature for the fit (20000 K)  
    diag (False)
        
    Returns BBtemp, BBtemperr
       
    """
    
    import astropy.units as ur
    from astropy.modeling import models
    from astropy.modeling.blackbody import FLAM
    from scipy.optimize import curve_fit
    import numpy as np
    
    bandone = kwargs.pop('bandone', [180,220]*ur.nm)
    bandtwo = kwargs.pop('bandtwo', [260,300]*ur.nm)
    magone = kwargs.pop('magone', 22*ur.ABmag)
    magtwo = kwargs.pop('magtwo', 22*ur.ABmag)
    magone_err = kwargs.pop('magone_err', 0.1*ur.ABmag)
    magtwo_err = kwargs.pop('magtwo_err', 0.1*ur.ABmag)
    
    bbtemp_init = kwargs.pop('bbtemp_init', 20000.) # Kelvin
    bolflux_init = 1.E-10 # erg/(cm**2 * s)
    
    # Since the fitter doesn't like quantities, make sure all inputs are in the correct units
    bandone_nm = bandone.to(ur.nm)
    bandtwo_nm = bandtwo.to(ur.nm)
    
    # Get central wavelengths (can be replaced later with effective wavelengths)
    wav = [np.mean(bandone_nm).value, np.mean(bandtwo_nm).value]
    
    # Lists of magnitudes are weird...
    mags = [magone.value, magtwo.value]*ur.ABmag
    mags_err = np.array([magone_err.value, magtwo_err.value])
    
    # Convert magnitudes and errors to flux densities and remove units
    fden = mags.to(FLAM,equivalencies=ur.spectral_density(wav*ur.nm)).value
    snrs = 1./(10.**(mags_err/2.5) - 1.)
    fden_err = fden / snrs

    # Fit blackbody:
    coeff, var_matrix = curve_fit(bbfunc, wav, fden, p0=[bbtemp_init, bolflux_init], sigma=fden_err, absolute_sigma=True)
    perr = np.sqrt(np.diag(var_matrix))
    
    bbtemp = coeff[0]*ur.K
    bbtemp_err = perr[0]*ur.K
    
    if diag:
        print()
        print('Fit blackbody to ABmags in two bands')
        print('Blackbody temperature: {}'.format(bbtemp))
        print('Band one: {}'.format(bandone))
        print('Band two: {}'.format(bandtwo))
        print('ABmag band one: {}'.format(magone))
        print('ABmag error band one: {}'.format(magone_err))
        print('ABmag band two: {}'.format(magtwo))
        print('ABmag error band two: {}'.format(magtwo_err))
        
        print('Flux density band one: {}'.format(fden[0]))
        print('Flux density band two: {}'.format(fden[1]))
        print('Flux density error band one: {}'.format(fden_err[0]))
        print('Flux density error band two: {}'.format(fden_err[1]))
        print('Fitted blackbody temperature: {}'.format(bbtemp))
        print('Fitted blackbody temperature error: {}'.format(bbtemp_err))
        print('')    
    
    if val:
        return bbtemp.value, bbtemp_err.value
    else:
        return bbtemp, bbtemp_err


def bb_abmag_fluence(val=False, **kwargs):
    """
    Take a blackbody with a certain temperature and convert to photon rate in a band.
    
    Now applies the various filters and returns *photon fluence* in the band
    
    Now also accepts bolometric flux as input.
    
    Inputs (defaults):
    umag = apparent u-band AB magnitude (22*ur.ABmag)
    swiftmag = apparent Swift UVW2 magnitude (22*ur.ABmag)
    ref = band to use for reference magnitude; options are 'u', 'swift' ('swift')
    bandone = Bandpass 1st filter (180-220)*ur.nm
    bandtwo = Bandpass 2nd filter (260-300)*ur.nm
    bbtemp = Blackbody temperature (20000*ur.K)
    dist = Distance (10*ur.pc)
    val = Return values without unit if True (False)
    bolflux = Bolometric flux; if not 1, refmag and distance are ignored (1*ur.erg/ur.cm**2/ur.s)
    
    diag (False)
        
    Returns ABmag1, ABmag2
       
    Also still to do: Add background from galaxies.
       
    """
    
    import astropy.units as ur
    import astropy.constants as cr
    from astropy.modeling import models
    from astropy.modeling.blackbody import FLAM
    import numpy as np
    from tdsat_telescope import apply_filters

    
    
    bbtemp = kwargs.pop('bbtemp', 20000.*ur.K)
#    bandone = kwargs.pop('bandone', [180,220]*ur.nm)
#    bandtwo = kwargs.pop('bandtwo', [260,300]*ur.nm)
    umag = kwargs.pop('umag', 22*ur.ABmag)
    swiftmag = kwargs.pop('swiftmag', 22*ur.ABmag)
    dist = kwargs.pop('dist', 10*ur.pc)
    ref = kwargs.pop('ref', 'swift')
    diag=kwargs.pop('diag', False)
    dist0 = 10*ur.pc
    bolflux = kwargs.pop('bolflux', 1.*ur.erg/(ur.cm**2 * ur.s))
    
    bandu = [340,380]*ur.nm # For comparison purposes
    bandsw = [172.53,233.57]*ur.nm # Swift UVW2 effective band (lambda_eff +/- 0.5 width_eff)

    wav = np.arange(1000,9000) * ur.AA # Wavelength scale in 1 Angstrom steps
    bb = models.BlackBody1D(temperature=bbtemp,bolometric_flux=bolflux) # Load the blackbody model
    flux = bb(wav).to(FLAM, ur.spectral_density(wav))

    # Get Swift reference AB mag
    fluxden_sw = np.mean(flux[(wav >= bandsw[0].to(ur.AA)) & (wav <= bandsw[1].to(ur.AA))])
    magsw = fluxden_sw.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandsw)))


    # Conver to flux AB mags across the band.
    flux_ab = flux.to(ur.ABmag, equivalencies = ur.spectral_density(wav))

    # Distance modulus
    distmod = (5*np.log10(dist/dist0)).value*ur.mag

    # Set up input:
    magoff = swiftmag - magsw

    # Apply the distance modulus and the Swift reference offset
    if (bolflux == 1.*ur.erg/(ur.cm**2 * ur.s)):
        flux_mag = flux_ab + magoff + distmod
    else:
        flux_mag = flux_ab

    # Convert back to flux
    flux_conv = flux_mag.to(FLAM, equivalencies=ur.spectral_density(wav))
    dw = 1*ur.AA
    ph_energy = (cr.h.cgs * cr.c.cgs / wav.cgs)

    # Convert to photon flux.
    ph_flux = flux_conv * dw / ph_energy

    # Apply filters, QE, etc.
    band1_fluence = apply_filters(wav, ph_flux, diag=diag, **kwargs).sum().sum()
    band2_fluence = apply_filters(wav, ph_flux, band = 2, diag=diag, **kwargs).sum()


    if diag:
        print()
        print('Compute ABmags in TD bands for blackbody')
        print('Blackbody temperature: {}'.format(bbtemp))
        print('Reference UVW2-band magnitude: {}'.format(swiftmag))
        print('Distance: {} (Reference distance is 10 pc)'.format(dist))
        print()
        print('Flux density Swift: {}'.format(fluxden_sw))
        print('Distance modulus: {}'.format(distmod))
        print('Raw ABmag Swift: {}'.format(magsw))
        print('Offset from Swift band: {}'.format(magoff))
        print('Fluence band one: {}'.format(band1_fluence))
        print('Fluence band two: {}'.format(band2_fluence))
        print('')    
    
    if val:
        return band1_fluence.value, band2_fluence.value
    else:
        return band1_fluence, band2_fluence


