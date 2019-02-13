def load_zodi(**kwargs):
    """
    From here
    https://cads.iiap.res.in/tools/zodiacalCalc/Documentation
    
    They provied the tabulated Zodi spectrum:
    https://cads.iiap.res.in/download/simulations/scaled_zodiacal_spec.txt
    
    From here
    Colina et al
    http://adsabs.harvard.edu/abs/1996AJ....112..307C
    
    According to this, the flux has been scaled so that at 5000 Ang
    the flux has units of 252 W / m2 / sr / micon 
    
    This takes as input the flux density as read off from Table 17 of 
    https://aas.aanda.org/articles/aas/pdf/1998/01/ds1449.pdf
    
    scale = scale in units of [1e-8 W / m2 / sr / micron at 500 nm]
    Default is for polar zodiacal emission, which is 77 in the above units.

    Toward the ecliptic plane this number can grow to be >1000 
    
    For a Sun avoidance of 45 degrees this looks like a value of 200 - 900
    based strongly on the heliocentric longitdue. However, if you try 
    72, 300, and 1000 it looks like you'll probably span this space.

    This returns 
    
    """
    from astropy import units as ur
    
    scale = kwargs.pop('scale', 77)
    
    ftab_unit = ur.W/ur.m**2 / ur.micron / ur.sr    

    spec = {'wavelength':[],
           'flux':[]}
    
    f = open('input_data/scaled_zodiacal_spec.txt', 'r')
    
    scale_norm = False
    for line in f:
        wave, flux = line.split()

        if not scale_norm:
            if float(wave) > 5e3:
                scale_norm = float(flux)
        
        spec['wavelength'].append(float(wave))
        spec['flux'].append(float(flux))
    f.close()
    
    # Normalize to flux density at 500 nm:
    spec['flux'] *= (scale*1e-8 / scale_norm) * ftab_unit
    
    return spec
    