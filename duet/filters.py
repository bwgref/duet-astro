def load_qe(**kwargs):
    """
    Loads the detector QE and returns the values.
    
    band = 1 (default, 180-220 nm)
    band = 2 (260-320 nm)
    band = 3 (340-380 nm)
    
    Syntax:
    
    wave, qe = load_qe(band = 1)
    
    """
    import astropy.units as ur
    import numpy as np
    
    band = kwargs.pop('band', 1)
    diag = kwargs.pop('diag', False)
    
    indir = 'input_data/'
    
    if band == 1:
        infile = indir+'detector_180_220nm.csv'
    if band == 2:
        infile = indir+'detector_260_300nm.csv'
    if band == 3:
        infile = indir+'detector_340_380nm.csv'
        
    f = open(infile, 'r')
    header = True
    qe = {}
    set = False
    for line in f:
        if header:
            header = False
            continue
        fields = line.split(',')
        if not set:
            wave = float(fields[0])
            qe = float(fields[3])
            set = True
        else:
            wave = np.append(wave, float(fields[0]))
            qe = np.append(qe, float(fields[3]))
 
    f.close()
    
    # Give wavelength a unit
    wave *= ur.nm
    
    if diag:
        print('Detector Q.E. loader')
        print('Band {} has input file {}'.format(band, infile))
        
    
    return wave, qe / 100.

def load_reflectivity(**kwargs):
    """
    Loads the optics reflectivity and returns the values.
    
    
    Syntax:
    
    wave, reflectivity = load_reflectivity()
    
    """
    import astropy.units as ur
    import numpy as np
    
    diag = kwargs.pop('diag', False)
    
    indir = 'input_data/'
    
    infile = indir+'al_mgf2_mirror_coatings.csv'

    f = open(infile, 'r')
    header = True
    qe = {}
    set = False
    for line in f:
        if header:
            header = False
            continue
        fields = line.split(',')
        if not set:
            wave = float(fields[0])
            reflectivity = float(fields[1])
            set = True
        else:
            wave = np.append(wave, float(fields[0]))
            reflectivity = np.append(reflectivity, float(fields[1]))
 
    f.close()
    
    # Give wavelength a unit
    wave *= ur.nm
    
    if diag:
        print('Optics reflectivity loader')
        print('Input file {}'.format(infile))
        
    
    return wave, reflectivity

def load_redfilter(**kwargs):
    """
    Loads the detector QE and returns the values.
    
    band = 1 (default, 180-220 nm)
    band = 2 (260-320 nm)
    
    Syntax:
    
    wave, transmission = load_redfilter(band=1)
    
    """
    import astropy.units as ur
    import numpy as np
    
    band = kwargs.pop('band', 1)
    diag = kwargs.pop('diag', False)
    light = kwargs.pop('light', True)
    
    indir = 'input_data/'
    
    if light:
        infile = indir+'duet{}_filter_light.csv'.format(band)
    else:
        infile = indir+'duet{}_filter.csv'.format(band)
        
        
    f = open(infile, 'r')
    header = True
    qe = {}
    set = False
    for line in f:
        if header:
            if (line.startswith('Wavelength')) or ('%T' in line):
                header = False
            continue
        fields = line.split(',')
        if not set:
            wave = float(fields[0])
            transmission = float(fields[1])
            set = True
        else:
            wave = np.append(wave, float(fields[0]))
            transmission = np.append(transmission, float(fields[1]))
 
    f.close()
    
    # Give wavelength a unit
    wave *= ur.nm
    
    if diag:
        print('Red filter loader')
        print('Band {} has input file {}'.format(band, infile))
        
    
    return wave, transmission / 100.


def apply_trans(wav_s, flux_s, wav_t, trans, **kwargs):
    """
    Interpolates transmission curve onto source wavelength array, applies transmission curve.
    
    Required inputs:
    wav_s (wavelength array of source spectrum, numpy array with units)
    flux_s (source spectrum, numpy array with units)
    wav_t (wavelength array of transmission curve, numpy array with units)
    trans (transmission curve, numpy array without units)
        
    Optional Inputs (defaults):
    diag = Diagnostics toggle (False)
    
    Returns corrected input spectrum
    
    """
    
    import astropy.units as ur
    import astropy.constants as cr
    import numpy as np
    
    # Check if the inputs make sense 
    assert len(wav_s) == len(flux_s), "Input spectrum and corresponding wavelength array are not the same length"
    assert len(wav_s) == len(flux_s), "Input spectrum and corresponding wavelength array are not the same length"   
    assert len(wav_t) == len(trans), "Transmission curve and corresponding wavelength array are not the same length"
    assert max(trans) <= 1., "Values larger than 1 found in transmission curve"
    assert min(trans) >= 0., "Values smaller than 0 found in transmission curve"
    assert wav_s.unit.is_equivalent(ur.m), "Input wavelength array does not have unit of length"
    assert wav_t.unit.is_equivalent(ur.m), "Transmission wavelength array does not have unit of length"
        
    # Make sure wav_s and wav_t are in the same units for interpolation
    wav_ttrue = wav_t.to(wav_s.unit)
    
    # Interpolate transmission curve onto source wavelength array
    trans_int = np.interp(wav_s,wav_ttrue,trans)

    # Correct input spectrum:
    flux_corr = flux_s * trans_int
    
    diag = kwargs.pop('diag', False)
    
    if diag:
        print('Diagnostics?')
    
    return flux_corr
    
def apply_filters(wave, spec, **kwargs):
    """
    Loads the detector QE and returns the values.
    
    band = 1 (default, 180-220 nm)
    band = 2 (260-320 nm)
    
    Syntax:
    
    wave, transmission = load_redfilter(band=1)
    
    """
    
    # Load filters
    ref_wave, reflectivity = load_reflectivity(**kwargs)
    qe_wave, qe = load_qe(**kwargs) 
    red_wave, red_trans = load_redfilter(**kwargs)    

    ref_flux = apply_trans(wave, spec, ref_wave, reflectivity/100.)
    qe_flux = apply_trans(wave, ref_flux, qe_wave, qe)
    band_flux = apply_trans(wave, qe_flux, red_wave, red_trans)

    return band_flux
