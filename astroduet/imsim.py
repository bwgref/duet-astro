import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astroduet.bbmag import bb_abmag_fluence
from astroduet.image_utils import construct_image, find, ap_phot, run_daophot, estimate_background
from astroduet.config import Telescope
from astroduet.background import background_pixel_rate
from astroduet.utils import duet_abmag_to_fluence
from astropy.table import Table
from astropy.io import fits
from astroduet.diff_image import py_zogy

def imsim(**kwargs):
    """
    Simulate images of sources in galaxies with a range of surface brightnesses
    Warning! Takes several hours to run on a laptop!
    
    Currently set up to run on atlas.
    
    Parameters
    ----------   
    tel: 'config' default is 'baseline'
        Sets telescope parameters
        
    run: string (date, as in '050719', or other identifying string)
        To track runs
    
    gal: 'spiral', 'elliptical' or 'dwarf', default is spiral
        Sets Sersic index and size
     
    zodi: 'low', 'med' or 'high', default is low 
        Use the medium zodiacal background rate. Overrides low_zodi.
        
    sfb: [sfb_low, sfb_high], default is [20,30] 
        List of lowest and highest surface brightness to simulate
    
    nmags: int, default is 70
        Number of source magnitudes to simulate, in 0.1 mag steps with sfb at the center, for each sfb
        
    nsrc: int, default is 100
        Number of sources to simulate at each source mag
    
    nref: list, default is [1,3,5,8]
    List of reference image depths
    
    Returns
    -------
    telescope_band_gal_sfb_zodi_src-mag.fits: fits file with simulated images. 
    """
    
    # Deal with kwargs:
    tel = kwargs.pop('tel', 'baseline')
    gal = kwargs.pop('gal', 'spiral')
    zodi = kwargs.pop('zodi', 'low')
    sfb_lim = kwargs.pop('sfb', [20,30])
    nmags = kwargs.pop('nmags', 70)
    nsrc = kwargs.pop('nsrc', 100)
    date = kwargs.pop('run') 
    ref_arr = kwargs.pop('nref',[1,3,5,8])

    # set some telescope, instrument parameters
    duet = Telescope(config=tel) 
    
    # Write telescope definition file
    teldef = duet.info()
    teldef_file = open('/Users/duetsim/duet-sims/image_library/run_'+date+'/teldef', 'w+')
    teldef_file.write(teldef)
    teldef_file.close()
            
    # Define image simulation parameters
    exposure = 300 * u.s
    frame = np.array([30,30]) # Dimensions of the image I'm simulating in DUET pixels (30x30 ~ 3x3 arcmin)
    psf_fwhm_pix = duet.psf_fwhm / duet.pixel
    
    # Get backgrounds
    if zodi == 'low':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, low_zodi = True, diag=False)
    elif zodi == 'med':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, med_zodi = True, diag=False)
    elif zodi == 'high':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, high_zodi = True, diag=False)
    
    # Define galaxy: amplitude is placeholder. Sizes are typical at 100 Mpc
    if gal == 'spiral':
        reff = 16.5 *u.arcsec
        gal_params = {'amplitude': 1,'r_eff': reff/(duet.pixel/6),'n':1, 'x_0': 0, 'y_0': 0}
    elif gal == 'elliptical':
        reff = 12.5 *u.arcsec
        gal_params = {'amplitude': 1,'r_eff': reff/(duet.pixel/6),'n':4, 'x_0': 0, 'y_0': 0}
    elif gal == 'dwarf':
        reff = 7 *u.arcsec
        gal_params = {'amplitude': 1,'r_eff': reff/(duet.pixel/6),'n':1, 'x_0': 0, 'y_0': 0}
    
    # Make galaxy surface brightness array
    sfb_arr = np.arange(sfb_lim[0],sfb_lim[1]+1.) # Now in steps of 1 mag
    
    # Make srcmag array:
    srcmag_arr = np.linspace(20.5 - 0.5*nmags*0.1, 20.5 + (0.5*nmags + 1)*0.1, num=nmags, endpoint=False) # Currently in steps of 0.1 mag
    
    # First DUET1
    for i, sfb in enumerate(sfb_arr):
        print('DUET1: Surface brightness level '+str(i+1)+' of '+str(len(sfb_arr))+'...')
        # Calculate count rate:
        surface_rate = duet.fluence_to_rate(duet_abmag_to_fluence(sfb*u.ABmag,duet.bandpass1)) # surface count rate at r_eff
        gal_params['amplitude'] = surface_rate.value * (duet.pixel.value/6)**2 # surface brightness (per pixel)
        # Make reference images:
        print('Building reference images...')
        empty_hdu = fits.PrimaryHDU()
        ref_hdu = fits.HDUList([empty_hdu])
        ref_hdu = update_header(ref_hdu, im_type='reference', zodi=zodi, gal=gal, sfb=sfb, reff=reff.value, 
                                band='DUET1', nframes=len(ref_arr), exptime=exposure.value)
        for nref in ref_arr:
            image = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=None,
                            sky_rate=bgd_band1, n_exp=nref)
            imhdu = fits.ImageHDU(image.value)
            imhdu.header['NFRAMES'] = (nref, 'Number of frames in reference image')
            imhdu.header['BUNIT'] = image.unit.to_string()
            imhdu.header['EXPTIME'] = (nref*exposure.value, 'Total exposure time of reference image (s)')
            ref_hdu.append(imhdu)
        # Write file    
        path  = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_'+gal+'/zodi_'+zodi+'/duet1/'
        filename = date+'_duet1_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_reference.fits'
        ref_hdu.writeto(path+filename, overwrite=True)
        #ref_hdu.writeto('data/test_ref.fits',overwrite=True)
        
        # Make source images:
        print('Building source images...')
        #srcmag_arr = np.arange(sfb - 0.5*nmags*0.1, sfb + (0.5*nmags + 1)*0.1, step=0.1) # Currently in steps of 0.1 mag
        gal_sizex = 2*reff/duet.pixel/frame[0]
        gal_startx = 0.5-gal_sizex
        gal_sizey = 2*reff/duet.pixel/frame[1]
        gal_starty = 0.5-gal_sizey
        for srcmag in srcmag_arr:
            empty_hdu = fits.PrimaryHDU()
            src_hdu = fits.HDUList([empty_hdu])
            src_hdu = update_header(src_hdu, im_type='source', zodi=zodi, gal=gal, sfb=sfb, reff=reff.value, 
                                    band='DUET1', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
            src_fluence = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, duet.bandpass1))
            for i in range(nsrc):
                source_loc = np.array([gal_startx+2*gal_sizex*np.random.random(), gal_starty+2*gal_sizey*np.random.random()])
                image = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=src_fluence,
                            source_loc=source_loc, sky_rate=bgd_band1, n_exp=1)
                imhdu = fits.ImageHDU(image.value)
                imhdu.header['SRC_POSX'] = (source_loc[0]*frame[0], 'X-position of source in image (pixels)')
                imhdu.header['SRC_POSY'] = (source_loc[1]*frame[1], 'Y-position of source in image (pixels)')
                imhdu.header['BUNIT'] = image.unit.to_string()
                imhdu.header['EXPTIME'] = (exposure.value, 'Exposure time (s)')
                src_hdu.append(imhdu)
            # Write file
            path = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_'+gal+'/zodi_'+zodi+'/duet1/'
            filename = date+'_duet1_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
            src_hdu.writeto(path+filename, overwrite=True)
            #src_hdu.writeto('test_im.fits',overwrite=True)

    # Same for DUET2
    for i, sfb in enumerate(sfb_arr):
        print('DUET2: Surface brightness level '+str(i+1)+' of '+str(len(sfb_arr))+'...')
        # Calculate count rate:
        surface_rate = duet.fluence_to_rate(duet_abmag_to_fluence(sfb*u.ABmag,duet.bandpass2)) # surface count rate at r_eff
        gal_params['amplitude'] = surface_rate.value * (duet.pixel.value/6)**2 # surface brightness (per pixel)
        # Make reference images:
        print('Building reference images...')
        empty_hdu = fits.PrimaryHDU()
        ref_hdu = fits.HDUList([empty_hdu])
        ref_hdu = update_header(ref_hdu, im_type='reference', zodi=zodi, gal=gal, sfb=sfb, reff=reff.value, 
                                band='DUET2', nframes=len(ref_arr), exptime=exposure.value)
        for nref in ref_arr:
            image = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=None,
                            sky_rate=bgd_band2, n_exp=nref)
            imhdu = fits.ImageHDU(image.value)
            imhdu.header['NFRAMES'] = (nref, 'Number of frames in reference image')
            imhdu.header['BUNIT'] = image.unit.to_string()
            imhdu.header['EXPTIME'] = (nref*exposure.value, 'Total exposure time of reference image (s)')
            ref_hdu.append(imhdu)
        # Write file    
        path = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_'+gal+'/zodi_'+zodi+'/duet2/'
        filename = date+'_duet2_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_reference.fits'
        ref_hdu.writeto(path+filename, overwrite=True)
        #ref_hdu.writeto('data/test_ref.fits',overwrite=True)
        
        # Make source images:
        print('Building source images...')
        #srcmag_arr = np.arange(sfb - 0.5*nmags*0.1, sfb + (0.5*nmags + 1)*0.1, step=0.1) # Currently in steps of 0.1 mag
        gal_sizex = 2*reff/duet.pixel/frame[0]
        gal_startx = 0.5-gal_sizex
        gal_sizey = 2*reff/duet.pixel/frame[1]
        gal_starty = 0.5-gal_sizey
        for srcmag in srcmag_arr:
            empty_hdu = fits.PrimaryHDU()
            src_hdu = fits.HDUList([empty_hdu])
            src_hdu = update_header(src_hdu, im_type='source', zodi=zodi, gal=gal, sfb=sfb, reff=reff.value, 
                                    band='DUET2', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
            src_fluence = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, duet.bandpass2))
            for i in range(nsrc):
                source_loc = np.array([gal_startx+2*gal_sizex*np.random.random(), gal_starty+2*gal_sizey*np.random.random()])
                image = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=src_fluence,
                            source_loc=source_loc, sky_rate=bgd_band2, n_exp=1)
                imhdu = fits.ImageHDU(image.value)
                imhdu.header['SRC_POSX'] = (source_loc[0]*frame[0], 'X-position of source in image (pixels)')
                imhdu.header['SRC_POSY'] = (source_loc[1]*frame[1], 'Y-position of source in image (pixels)')
                imhdu.header['BUNIT'] = image.unit.to_string()
                imhdu.header['EXPTIME'] = (exposure.value, 'Exposure time (s)')
                src_hdu.append(imhdu)
            # Write file
            path = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_'+gal+'/zodi_'+zodi+'/duet2/'
            filename = date+'_duet2_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
            src_hdu.writeto(path+filename, overwrite=True)
            #src_hdu.writeto('test_im.fits',overwrite=True)
                
def imsim_no_gal(**kwargs):
    """
    Simulate images of sources without background galaxies.
    Warning! Takes several hours to run on a laptop!
    
    Currently set up to run on atlas.
    
    Parameters
    ----------   
    tel: 'config' default is 'baseline'
        Sets telescope parameters
        
    run: string (date, as in '050719')
        To track runs
        
    zodi: 'low', 'med' or 'high', default is low 
        Use the medium zodiacal background rate. Overrides low_zodi.
        
    nmags: int, default is 70
        Number of source magnitudes to simulate, in 0.1 mag steps around 20.5. For the default 70, source magnitudes range from 17 to 23.9
        
    nsrc: int, default is 100
        Number of sources to simulate at each source mag
        
    nref: list, default is [1,3,5,8]
        List of reference image depths
        
    Returns
    -------
    telescope_band_zodi_src-mag.fits: fits file with simulated images. 
    """
    
    # Deal with kwargs:
    tel = kwargs.pop('tel', 'baseline')
    zodi = kwargs.pop('zodi', 'low')
    nmags = kwargs.pop('nmags', 70)
    nsrc = kwargs.pop('nsrc', 100)
    date = kwargs.pop('run') 
    ref_arr = kwargs.pop('nref',[1,3,5,8])

    # set some telescope, instrument parameters
    duet = Telescope(config=tel) 
    
    # Write telescope definition file
    teldef = duet.info()
    teldef_file = open('/Users/duetsim/duet-sims/image_library/run_'+date+'/teldef', 'w+')
    teldef_file.write(teldef)
    teldef_file.close()
            
    # Define image simulation parameters
    exposure = 300 * u.s
    frame = np.array([30,30]) # Dimensions of the image I'm simulating in DUET pixels (30x30 ~ 3x3 arcmin)
    psf_fwhm_pix = duet.psf_fwhm / duet.pixel
    
    # Get backgrounds
    if zodi == 'low':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, low_zodi = True, diag=False)
    elif zodi == 'med':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, med_zodi = True, diag=False)
    elif zodi == 'high':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, high_zodi = True, diag=False)
    
    # Make srcmag array:
    srcmag_arr = np.linspace(20.5 - 0.5*nmags*0.1, 20.5 + (0.5*nmags + 1)*0.1, num=nmags, endpoint=False) # Currently in steps of 0.1 mag
    
    # First DUET1
    print('DUET1...')
    # Make reference images:
    print('Building reference images...')
    empty_hdu = fits.PrimaryHDU()
    ref_hdu = fits.HDUList([empty_hdu])
    ref_hdu = update_header(ref_hdu, im_type='reference', zodi=zodi, gal='None', reff=reff.value, 
                            band='DUET1', nframes=len(ref_arr), exptime=exposure.value)
    for nref in ref_arr:
        image = construct_image(frame, exposure, gal_type=None, source=None,
                        sky_rate=bgd_band1, n_exp=nref)
        imhdu = fits.ImageHDU(image.value)
        imhdu.header['NFRAMES'] = (nref, 'Number of frames in reference image')
        imhdu.header['BUNIT'] = image.unit.to_string()
        imhdu.header['EXPTIME'] = (nref*exposure.value, 'Total exposure time of reference image (s)')
        ref_hdu.append(imhdu)
    # Write file    
    path  = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_none/zodi_'+zodi+'/duet1/'
    filename = date+'_duet1_zodi-'+zodi+'_reference.fits'
    ref_hdu.writeto(path+filename, overwrite=True)
    #ref_hdu.writeto('data/test_ref.fits',overwrite=True)
    
    # Make source images:
    print('Building source images...')
    for srcmag in srcmag_arr:
        empty_hdu = fits.PrimaryHDU()
        src_hdu = fits.HDUList([empty_hdu])
        src_hdu = update_header(src_hdu, im_type='source', zodi=zodi, gal='None', reff=reff.value, 
                                band='DUET1', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
        src_fluence = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, duet.bandpass1))
        for i in range(nsrc):
            source_loc = np.array([np.random.random(), np.random.random()])
            image = construct_image(frame, exposure, gal_type=None, source=src_fluence,
                        source_loc=source_loc, sky_rate=bgd_band1, n_exp=1)
            imhdu = fits.ImageHDU(image.value)
            imhdu.header['SRC_POSX'] = (source_loc[0]*frame[0], 'X-position of source in image (pixels)')
            imhdu.header['SRC_POSY'] = (source_loc[1]*frame[1], 'Y-position of source in image (pixels)')
            imhdu.header['BUNIT'] = image.unit.to_string()
            imhdu.header['EXPTIME'] = (exposure.value, 'Exposure time (s)')
            src_hdu.append(imhdu)
        # Write file
        path = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_none/zodi_'+zodi+'/duet1/'
        filename = date+'_duet1_zodi-'+zodi+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
        src_hdu.writeto(path+filename, overwrite=True)

    # Now DUET2
    print('DUET2...')
    # Make reference images:
    print('Building reference images...')
    empty_hdu = fits.PrimaryHDU()
    ref_hdu = fits.HDUList([empty_hdu])
    ref_hdu = update_header(ref_hdu, im_type='reference', zodi=zodi, gal='None', reff=reff.value, 
                            band='DUET2', nframes=len(ref_arr), exptime=exposure.value)
    for nref in ref_arr:
        image = construct_image(frame, exposure, gal_type=None, source=None,
                        sky_rate=bgd_band2, n_exp=nref)
        imhdu = fits.ImageHDU(image.value)
        imhdu.header['NFRAMES'] = (nref, 'Number of frames in reference image')
        imhdu.header['BUNIT'] = image.unit.to_string()
        imhdu.header['EXPTIME'] = (nref*exposure.value, 'Total exposure time of reference image (s)')
        ref_hdu.append(imhdu)
    # Write file    
    path  = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_none/zodi_'+zodi+'/duet2/'
    filename = date+'_duet2_zodi-'+zodi+'_reference.fits'
    ref_hdu.writeto(path+filename, overwrite=True)
    #ref_hdu.writeto('data/test_ref.fits',overwrite=True)
    
    # Make source images:
    print('Building source images...')
    for srcmag in srcmag_arr:
        empty_hdu = fits.PrimaryHDU()
        src_hdu = fits.HDUList([empty_hdu])
        src_hdu = update_header(src_hdu, im_type='source', zodi=zodi, gal='None', reff=reff.value, 
                                band='DUET2', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
        src_fluence = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, duet.bandpass2))
        for i in range(nsrc):
            source_loc = np.array([np.random.random(), np.random.random()])
            image = construct_image(frame, exposure, gal_type=None, source=src_fluence,
                        source_loc=source_loc, sky_rate=bgd_band2, n_exp=1)
            imhdu = fits.ImageHDU(image.value)
            imhdu.header['SRC_POSX'] = (source_loc[0]*frame[0], 'X-position of source in image (pixels)')
            imhdu.header['SRC_POSY'] = (source_loc[1]*frame[1], 'Y-position of source in image (pixels)')
            imhdu.header['BUNIT'] = image.unit.to_string()
            imhdu.header['EXPTIME'] = (exposure.value, 'Exposure time (s)')
            src_hdu.append(imhdu)
        # Write file
        path = '/Users/duetsim/duet-sims/image_library/run_'+date+'/gal_none/zodi_'+zodi+'/duet2/'
        filename = date+'_duet2_zodi-'+zodi+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
        src_hdu.writeto(path+filename, overwrite=True)
 
def update_header(hdu, **kwargs):
    """
    Update fits header for simulated images.
    
    Parameters
    ----------
    hdu: HDU object
        The HDU extension that you want to update

    Other keywords (default = None)
    --------------
    im_type: string
        Type of images in fits file ('reference' or 'source')
        
    zodi: string
        Setting of zodiacal background ('low', 'med', 'high')
        
    gal: string
        Galaxy morphology ('spiral', 'elliptical', 'dwarf', 'None')

    sfb: float
        Surface brightness at effective radius in ABmag/arcsec^2
        
    reff: float
        Effective radius in arcseconds
        
    srcmag: float
        Source magnitude in ABmag
               
    band: string
        DUET band ('DUET1', 'DUET2')
        
    nframes: float
        Number of image extensions
        
    exptime: float
        Exposure time in seconds
    
    Returns
    -------
    Updated HDU object
    """

    im_type = kwargs.pop('im_type', None)
    zodi = kwargs.pop('zodi', None)
    gal = kwargs.pop('gal', None)
    sfb = kwargs.pop('sfb', None)
    reff = kwargs.pop('reff', None)
    srcmag = kwargs.pop('srcmag', None)
    band = kwargs.pop('band', None)
    nframes = kwargs.pop('nframes', None)
    exptime = kwargs.pop('exptime', None)
    
    head = hdu[0].header
    head['NEXTEND'] = nframes
    head['FILETYPE'] = (im_type, 'type of images in fits file')
    head['DUETBAND'] = band
    head['ITIME'] = (exptime, 'Exposure time per frame (seconds)')
    head['GALTYPE'] = (gal, 'galaxy morphology')
    head['SFB_EFF'] = (sfb, 'surface brighness at R_eff (ABmag/arcsec^2)') 
    head['R_EFF'] = (reff, 'galaxy effective radius (arcsec)')
    head['ZODI_LEV'] = (zodi, 'Zodiacal background level')
    if im_type == 'source':
        head['SRC_MAG'] = (srcmag, 'source magnitude (ABmag)')
    
    return hdu
    
def run_srcdetect(run='050719',gal='spiral',zodi='low',band='duet1'):
    """
    Run background estimation, image differencing and source detection on simulated images
    
    Currently set up to run on atlas.
    
    gal = 'none' doesn't work yet!!
    
    Parameters
    ----------   
    run: string (date, as in '050719')
        To track runs
        
    zodi: 'low', 'med' or 'high', default is low 
        
    band: 'duet1' or 'duet2'
    
    gal: 'spiral', 'elliptical', 'dwarf' or 'none'
            
    Returns
    -------
    telescope_band_zodi_src-mag.fits: fits file with simulated images. 
    """
    
    # Initialize parameters
    duet = Telescope()
    # PSF stuff
    oversample = 5
    pixel_size_init = duet.pixel / oversample
    psf_model = duet.psf_model(pixel_size=pixel_size_init, x_size=25, y_size=25)
    psf_os = psf_model.array
    shape = (5, 5, 5, 5)
    psf_array = psf_os.reshape(shape).sum(-1).sum(1)
    psf_fwhm_pix = duet.psf_fwhm / duet.pixel

    # Set up path
    path = '/Users/duetsim/duet-sims/image_library/run_'+run+'/gal_'+gal+'/zodi_'+zodi+'/'+band+'/'
    
    if gal == 'spiral':
        sfb_arr = np.arange(20.,30.).astype(str)
    elif gal == 'elliptical':
        sfb_arr = np.arange(21.,31.).astype(str)
    
    if band == 'duet1':
        bandpass = duet.bandpass1
    elif band == 'duet2':
        bandpass = duet.bandpass2
        
    src_arr = np.linspace(17.0, 23.9, num=70)
    
    # Set up results table
    # columns: galaxy mag, source input mag, source input count rate, distance from galaxy center, reference depth, source detected True/False, 
    # if True: retrieved count rate, count rate error; number of false positives
    tab = Table(np.zeros(9), names=('galmag', 'srcmag', 'src-ctrate', 'dist', 'ref_depth', 'detected',
                                                    'ctrate', 'ctrate_err', 'false-pos'), dtype=('f8','f8','f8','f8',
                                                    'i8','b','f8','f8','i8'), meta={'name': gal+' - '+zodi+ 'zodi - '+band})
    
    for sfb in sfb_arr:
        print('SFB: '+sfb)
        reffile = run+'_'+band+'_'+gal+'_'+sfb+'_zodi-'+zodi+'_reference.fits'
        hdu_ref = fits.open(path+reffile)
        
        for i in range(hdu_ref[0].header['NEXTEND']):
            # Prepare reference image:
            ref_image_rate = hdu_ref[i+1].data / hdu_ref[i+1].header['EXPTIME'] *u.ph / u.s
            ref_bkg, ref_bkg_rms_median = estimate_background(ref_image_rate, method='1D', sigma=2)
            ref_rate_bkgsub = ref_image_rate - ref_bkg
            s_r = np.sqrt(ref_image_rate)
            sr = np.mean(s_r)
            # Get depth of reference image
            ref_depth = hdu_ref[i+1].header['NFRAMES']
            
            for srcmag in src_arr:
                imfile = run+'_'+band+'_'+gal+'_'+sfb+'_zodi-'+zodi+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
                hdu_im = fits.open(path+imfile)
                # Get input countrate
                src_ctrate = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, bandpass))
                for j in range(hdu_im[0].header['NEXTEND']):
                    # Get source distance from center of galaxy
                    dist = np.sqrt((14.5-hdu_im[j+1].header['SRC_POSX'])**2 + (14.5-hdu_im[j+1].header['SRC_POSY'])**2) * duet.pixel
                    # Prepare science image:
                    image_rate = hdu_im[j+1].data / hdu_im[j+1].header['EXPTIME'] *u.ph / u.s
                    image_bkg, image_bkg_rms_median = estimate_background(image_rate, method='1D', sigma=2)
                    image_rate_bkgsub = image_rate - image_bkg
                    s_n = np.sqrt(image_rate)
                    sn = np.mean(s_n)
                    
                    dx, dy = 0.1, 0.01 # Astrometric uncertainty (sigma)
                    # Run zogy:
                    diff_image, d_psf, s_corr = py_zogy(image_rate_bkgsub.value,
                                        ref_rate_bkgsub.value,
                                        psf_array,psf_array,
                                        s_n.value,s_r.value,
                                        sn.value,sr.value,dx,dy)
        
                    diff_image *= image_rate_bkgsub.unit
                    # Find sources:
                    star_tbl, bkg_image, threshold = find(diff_image,psf_fwhm_pix.value,method='peaks')
                    
                    # Define separation from input source and find nearest peak:
                    if len(star_tbl) > 0:
                        sep = np.sqrt((star_tbl['x'] - hdu_im[j+1].header['SRC_POSX'])**2 + (star_tbl['y'] - hdu_im[j+1].header['SRC_POSY'])**2)
                        src = np.argmin(sep)
                        if sep[src] < 1.5:
                            detected = True
                            # Run aperture photometry
                            result, apertures, annulus_apertures = ap_phot(diff_image,star_tbl[src],duet.read_noise,hdu_im[j+1].header['EXPTIME']*u.s)
                            ctrate, ctrate_err = result['aper_sum_bkgsub'],result['aperture_sum_err']
                            fp = len(star_tbl) - 1
                        else:
                            detected = False
                            ctrate, ctrate_err = np.nan, np.nan
                            fp = len(star_tbl)
                    else:
                        detected = False
                        ctrate, ctrate_err = np.nan, np.nan
                        fp = len(star_tbl)
                        
                    tab.add_row([float(sfb), srcmag, src_ctrate, dist, ref_depth, detected,
                                                       ctrate, ctrate_err, fp])
                    
                hdu_im.close()
        hdu_ref.close()
    tab.remove_row(0)
    
    # Save output table
    tab.write('run'+run+'_gal-'+gal+'_zodi-'+zodi+'-'+band+'.fits', format='fits', overwrite=True)