def bb_abmag(diag=False, val=False, **kwargs):
    """
    Take a blackbody with a certain temperature and convert to AB magnitudes in two bands.
    Scaled to u-band magnitude.
    
    Convert AB magnitude for a source into the number of source counts.
    
    Inputs (defaults):
    umag = apparent u-band AB magnitude (22*ur.ABmag)
    bandone = Bandpass 1st filter (180-220)*ur.nm
    bandtwo = Bandpass 2nd filter (260-300)*ur.nm
    bbtemp = Blackbody temperature (20000*ur.K)
    val = Return values without unit if True (False)
    
    diag (False)
        
    Returns ABmag1, ABmag2
    
    
       
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
    
    bandu = [340,380]*ur.nm # For comparison purposes

    wav = np.arange(1000,9000) * ur.AA # Wavelength scale in 1 Angstrom steps
    bb = models.BlackBody1D(temperature=bbtemp) # Load the blackbody model
    flux = bb(wav).to(FLAM, ur.spectral_density(wav))

    # Calculate mean flux density in each band:
    fluxden_one = np.mean(flux[(wav >= bandone[0].to(ur.AA)) & (wav <= bandone[1].to(ur.AA))])
    fluxden_two = np.mean(flux[(wav >= bandtwo[0].to(ur.AA)) & (wav <= bandtwo[1].to(ur.AA))])
    fluxden_u = np.mean(flux[(wav >= bandu[0].to(ur.AA)) & (wav <= bandu[1].to(ur.AA))])

    # Convert to AB magnitudes:
    magone = fluxden_one.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandone)))
    magtwo = fluxden_two.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandtwo)))
    magu = fluxden_u.to(ur.ABmag, equivalencies=ur.spectral_density(np.mean(bandu)))

    # Offset from comparison u-band magnitude:
    magoff = umag - magu
    
    # Apply offsets
    magone_final = magone + magoff
    magtwo_final = magtwo + magoff
 
    if diag:
        print()
        print('Compute ABmags in TD bands for blackbody')
        print('Blackbody temperature: {}'.format(bbtemp))
        print('Reference u-band magnitude: {}'.format(umag))
        print('Band one: {}'.format(bandone))
        print('Band two: {}'.format(bandtwo))
        
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
    
