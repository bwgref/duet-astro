def load_telescope_parameters(version, **kwargs):
    """
    Utility script to load the telescope parameters
    
    version = 0: Pre-design version (to compare with Rick's stuff)
    version = 1: 210 mm design
    version = 2: 300 mm design
    version = 3: 350 mm design
    version = 4: 400 mm design

    Syntax:
    diameter, qe, psf_fwhm, pixel_size, efficiency = load_telescope_parameters(version)
    
    2019/02/22 (BG)
    ---
        
    Note, going to depreicate versions < 4 eventually since those assume that
    the pixels are 0.5 * pixel size
    
    To be done: Remove QE from this method and get it somewhere else.
    ---
    
    """
    
    
    import astropy.units as ur
    from numpy import pi
    
    diag = kwargs.pop('diag', True)
    
    name = ''
    
    # Eventually depricate these things
    if version == 0:
        qe = 0.8 # To be improved later.
        diameter = 30*ur.cm
        psf_fwhm = 10*ur.arcsec
        pixel_size = psf_fwhm * 0.5
        efficiency = 0.87 # Ultrasat spec
    if version == 1:
        qe = 0.8
        efficiency = 0.54 # Reported from Mike  
        diameter = 21 * ur.cm
        psf_fwhm = 4 * ur.arcsec
        pixel_size = psf_fwhm * 0.5

    if version == 2:
        qe = 0.8
        efficiency = 0.65 # Reported from Mike
        diameter = 30 * ur.cm
        psf_fwhm = 9*ur.arcsec
        pixel_size = psf_fwhm * 0.5
    
    if version == 3:
        qe = 0.8
        diameter = 35*ur.cm
        efficiency = 0.67 # Reported from Mike
        psf_fwhm = 18*ur.arcsec
        pixel_size = psf_fwhm * 0.5

    if version == 4:
        qe = 0.8
        diameter = 40*ur.cm
        efficiency = 0.70 # Reported from Mike
        psf_fwhm = 23*ur.arcsec
        pixel_size = psf_fwhm * 0.5


    # Versions below here allow the PSF and the pixel to be decoupled

    # "Big Schmidt" w/ 6k x 6k array
    if version == 5:
        name = 'Big Schmidt'
        qe = 0.7
        diameter = 33.0*ur.cm
        eff_diam = 29.1*ur.cm
        efficiency = (eff_diam/diameter)**2

        plate_scale = 0.43 # arcsec per micron
        
        psf_fwhm_um = 21.6 # microns
        psf_fwhm = plate_scale * psf_fwhm_um * ur.arcsec
        
        pixel_size = plate_scale * 10 * ur.arcsec

    
    # Smaller Schmidts (same focal length?) each with 6k x 6k focal plane array
    if version == 6:
        name = 'Two mini Big Schmidts'
        qe = 0.7
        diameter = 21.0*ur.cm
        eff_diam = 15.1*ur.cm
        efficiency = (eff_diam/diameter)**2

        plate_scale = 0.43 # arcsec per micron
        
        psf_fwhm_um = 6.7 # microns
        psf_fwhm = plate_scale * psf_fwhm_um * ur.arcsec
        
        pixel_size = plate_scale * 10 * ur.arcsec

    
    # Medium Schmidt (same focal length?) each with 6k x 6k focal plane array
    if version == 7:
        name = 'Medium Schmidt'

        qe = 0.7
        diameter = 24.0*ur.cm
        eff_diam = 19.1*ur.cm
        efficiency = (eff_diam/diameter)**2

        plate_scale = 0.43 # arcsec per micron
        
        psf_fwhm_um = 7.6 # microns
        psf_fwhm = plate_scale * psf_fwhm_um * ur.arcsec
        
        pixel_size = plate_scale * 10 * ur.arcsec

    # Smaller Medium Schmidts (same focal length?) each with 6k x 6k focal plane array
    if version == 8:
        name = 'Two Small "Medium" Schmidts'
        qe = 0.7
        diameter = 14.0*ur.cm
        eff_diam = 6.3*ur.cm
        efficiency = (eff_diam/diameter)**2

        plate_scale = 0.43 # arcsec per micron
        
        psf_fwhm_um = 8.6 # microns
        psf_fwhm = plate_scale * psf_fwhm_um * ur.arcsec
        
        pixel_size = plate_scale * 10 * ur.arcsec

    # Fast Medium Schmidts (same focal length?) each with 6k x 6k focal plane array
    if version == 9:
        name = 'Fast Schmidt'

        qe = 0.7
        diameter = 32.0*ur.cm
        eff_diam = 29.89*ur.cm
        efficiency = (eff_diam/diameter)**2

        plate_scale = 0.64 # arcsec per micron
        
        psf_fwhm_um = 44.3 # microns
        psf_fwhm = plate_scale * psf_fwhm_um * ur.arcsec
        
        pixel_size = plate_scale * 10 * ur.arcsec

    # Fast Medium Schmidts (same focal length?) each with 6k x 6k focal plane array
    if version == 10:
        name="Mini Fast Schmidts"
        qe = 0.7
        diameter = 22.0*ur.cm
        eff_diam = 19.2*ur.cm
        efficiency = (eff_diam/diameter)**2

        plate_scale = 0.64 # arcsec per micron
        
        psf_fwhm_um = 14.1 # microns
        psf_fwhm = plate_scale * psf_fwhm_um * ur.arcsec
        
        pixel_size = plate_scale * 10 * ur.arcsec

    if diag:
        print('Telescope Configuration {}'.format(version))
        print('Name: {}'.format(name))
        print('Entrance Pupil diameter {}'.format(diameter))
        print('Optical Effifiency {}'.format(efficiency))
        print('PSF FWHM {}'.format(psf_fwhm))
        print('Pixel size {}'.format(pixel_size))
        print('Effective Aperture {}'.format(diameter*(efficiency)**0.5))
        print('Effective Area {}'.format( efficiency * pi * (0.5*diameter)**2))
              
    return diameter, qe, psf_fwhm, pixel_size, efficiency
    
    
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
        print('Input file {}'.format(band, infile))
        
    
    return wave, reflectivity

def make_o2_filter(in_wave,**kwargs):
    """
    Mocks up a filter function to remove the O2 line flux.
    and "rejection" out of band.
    
    Optional inputs (defaults):
    low_wave = lower wavelength of the band (180*ur.nm)
    high_wave = lower wavelength of the band (220*ur.nm)
    rejection = Out of band rejection level (1e-3)
    diag = Give diagnostic info (False)
    
    
    Syntax:
    filter = make_filter(wave)
        
    
    """
    import astropy.units as ur
    import numpy as np

    low_wave = kwargs.pop('low_wave', 2468*ur.AA)
    high_wave = kwargs.pop('high_wave', 2472*ur.AA) 
    rejection = kwargs.pop('rejection', 1e-3)
    diag = kwargs.pop('diag', False)

    
    wave = in_wave.to(ur.Angstrom).value

    low_check = low_wave.to(ur.Angstrom).value
    high_check = high_wave.to(ur.Angstrom).value
    

    red_filter = np.zeros_like(wave)
    red_filter[(wave<low_check) | (wave>high_check)] = 1.0
    red_filter[(wave>=low_check) & (wave<=high_check)] += rejection

    return red_filter


def make_red_filter(in_wave, **kwargs):
    """
    Mocks up a filter function at all of the "wave" points with 1 "in band"
    and "rejection" out of band.
    
    Optional inputs (defaults):
    low_wave = lower wavelength of the band (180*ur.nm)
    high_wave = lower wavelength of the band (220*ur.nm)
    rejection = Out of band rejection level (1e-3)
    diag = Give diagnostic info (False)
    
    
    Syntax:
    filter = make_filter(wave)
        
    
    """
    import astropy.units as ur
    import numpy as np
    
    low_wave = kwargs.pop('low_wave', 180*ur.nm)
    high_wave = kwargs.pop('high_wave', 220*ur.nm) 
    rejection = kwargs.pop('rejection', 1e-3)
    diag = kwargs.pop('diag', False)

    wave = in_wave.to(ur.Angstrom).value

    low_check = low_wave.to(ur.Angstrom).value
    high_check = high_wave.to(ur.Angstrom).value
    

    red_filter = np.zeros_like(wave)
    red_filter[(wave<low_check) | (wave>high_check)] += rejection
    red_filter[(wave>=low_check) & (wave<=high_check)] += 1.0
    
    if diag:
        print('Low wavelength: {}'.format(low_wave))
        print('High wavelength: {}'.format(high_wave))
        print('Rejection level: {}'.format(rejection))
    
    
    return red_filter
