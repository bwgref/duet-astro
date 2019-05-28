import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astroduet.bbmag import bb_abmag_fluence
from astroduet.image_utils import construct_image, find, ap_phot, run_daophot, estimate_background
from astroduet.config import Telescope
from astroduet.background import background_pixel_rate
from astroduet.utils import duet_abmag_to_fluence, duet_no_from_band
from astropy.table import Table
from astropy.io import fits
from astroduet.diff_image import py_zogy
import os.path

def imsim(**kwargs):
    """
    Simulate images of sources in galaxies with a range of surface brightnesses
    Warning! Takes several hours to run (~6 hours for 10 surface brightness levels, nmag=71 and nsrc=100)

    Will make a directory tree run/galaxy/zodi-level/duet[1,2] in directory where it's run.

    Parameters
    ----------
    tel: 'config' default is 'baseline'
        Sets telescope parameters

    run: string (date, as in '050719', or other identifying string)
        To track runs

    gal: 'spiral', 'elliptical', 'dwarf' or 'none', default is spiral
        Sets Sersic index and size

    zodi: 'low', 'med' or 'high', default is low
        Use the medium zodiacal background rate. Overrides low_zodi.

    sfb: [sfb_low, sfb_high], default is [20,30]
        List of lowest and highest surface brightness to simulate

    nmags: int, default is 71
        Number of source magnitudes to simulate, in 0.1 mag steps around 20.5

    nsrc: int, default is 100
        Number of sources to simulate at each source mag

    stack: int, default is 1
        Depth of science image (number of stacked 300s exposures)

    nref: list, default is [1,3,7,11]
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
    nmags = kwargs.pop('nmags', 71)
    nsrc = kwargs.pop('nsrc', 100)
    run = kwargs.pop('run')
    stack = kwargs.pop('stack', 1)
    ref_arr = kwargs.pop('nref',[1,3,7,11])

    # set some telescope, instrument parameters
    duet = Telescope(config=tel)

    # Set/make directories:
    path1  = sim_path(run=run, gal=gal, zodi=zodi, band='duet1')
    path2  = sim_path(run=run, gal=gal, zodi=zodi, band='duet2')

    # Write telescope definition file if it doesn't exist yet for this run:
    if not os.path.exists('run_'+run+'/teldef'):
        teldef = duet.info()
        teldef_file = open('run_'+run+'/teldef', 'w+')
        teldef_file.write(teldef)
        teldef_file.close()

    # Define image simulation parameters
    exposure = 300 * u.s
    frame = np.array([30,30]) # Dimensions of the image I'm simulating in DUET pixels (30x30 ~ 3x3 arcmin)
    oversample = 6 # Hardcoded in construct_image

    # Get backgrounds
    if zodi == 'low':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, low_zodi = True, diag=False)
    elif zodi == 'med':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, med_zodi = True, diag=False)
    elif zodi == 'high':
        [bgd_band1, bgd_band2] = background_pixel_rate(duet, high_zodi = True, diag=False)

    # Define galaxy: magnitude is placeholder. Sizes are typical at 100 Mpc
    if gal == 'spiral':
        reff = 16.5 *u.arcsec
        gal_params = {'magnitude': 1,'r_eff': reff/(duet.pixel/oversample),'n':1, 'x_0': 0, 'y_0': 0}
    elif gal == 'elliptical':
        reff = 12.5 *u.arcsec
        gal_params = {'magnitude': 1,'r_eff': reff/(duet.pixel/oversample),'n':4, 'x_0': 0, 'y_0': 0}
    elif gal == 'dwarf':
        reff = 7 *u.arcsec
        gal_params = {'magnitude': 1,'r_eff': reff/(duet.pixel/oversample),'n':1, 'x_0': 0, 'y_0': 0}
    elif gal == 'none':
        gal_params = None

    # Make galaxy surface brightness array (if necessary)
    if gal != 'none':
        sfb_arr = np.arange(sfb_lim[0],sfb_lim[1]+1.) # Now in steps of 1 mag

    # Make srcmag array:
    srcmag_arr = np.linspace(20.5 - 0.5*(nmags-1)*0.1, 20.5 + 0.5*(nmags-1)*0.1, num=nmags, endpoint=True) # Currently in steps of 0.1 mag

    # No background galaxy:
    if gal == 'none':
        # Make reference images:
        ref_hdu_1, ref_hdu_2 = run_sim_ref(duet=duet, bkg1=bgd_band1, bkg2=bgd_band2,
                                ref_arr=ref_arr, gal=False, exposure=exposure, frame=frame)
        # Update headers:
        ref_hdu_1 = update_header(ref_hdu_1, im_type='reference', zodi=zodi, gal=gal,
                                band='DUET1', nframes=len(ref_arr), exptime=exposure.value)
        ref_hdu_2 = update_header(ref_hdu_2, im_type='reference', zodi=zodi, gal=gal,
                                band='DUET2', nframes=len(ref_arr), exptime=exposure.value)

        # Write files
        ref_filename1 = run+'_duet1_zodi-'+zodi+'_reference.fits'
        ref_hdu_1.writeto(path1+'/'+ref_filename1, overwrite=True)

        ref_filename2 = run+'_duet2_zodi-'+zodi+'_reference.fits'
        ref_hdu_2.writeto(path2+'/'+ref_filename2, overwrite=True)

        # Make source images:
        for srcmag in srcmag_arr:
            src_hdu_1, src_hdu_2 = run_sim(duet=duet, bkg1=bgd_band1, bkg2=bgd_band2, stack=stack, 
                                srcmag=srcmag, nsrc=nsrc, gal=False, exposure=exposure, frame=frame)
            # Update headers
            src_hdu_1 = update_header(src_hdu_1, im_type='source', zodi=zodi, gal=gal,
                                band='DUET1', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
            src_hdu_2 = update_header(src_hdu_2, im_type='source', zodi=zodi, gal=gal,
                                band='DUET2', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
            
            # Write file
            filename1 = run+'_duet1_zodi-'+zodi+'_stack-'+str(stack)+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
            src_hdu_1.writeto(path1+'/'+filename1, overwrite=True)
            
            filename2 = run+'_duet2_zodi-'+zodi+'_stack-'+str(stack)+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
            src_hdu_2.writeto(path2+'/'+filename2, overwrite=True)

    # Yes background galaxy:
    else:
        for i, sfb in enumerate(sfb_arr):
            print('Surface brightness level '+str(i+1)+' of '+str(len(sfb_arr))+'...')
            # Calculate count rate:
            gal_params['magnitude'] = sfb
            # Make reference images:
            ref_hdu_1, ref_hdu_2 = run_sim_ref(duet=duet, bkg1=bgd_band1, bkg2=bgd_band2, 
                                    ref_arr=ref_arr, gal=True, gal_params=gal_params, exposure=exposure, frame=frame)
            # Update headers:
            ref_hdu_1 = update_header(ref_hdu_1, im_type='reference', zodi=zodi, gal=gal,
                                    band='DUET1', nframes=len(ref_arr), exptime=exposure.value)
            ref_hdu_2 = update_header(ref_hdu_2, im_type='reference', zodi=zodi, gal=gal,
                                    band='DUET2', nframes=len(ref_arr), exptime=exposure.value)
            # Write files:
            ref_filename1 = run+'_duet1_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_reference.fits'
            ref_hdu_1.writeto(path1+'/'+ref_filename1, overwrite=True)
            
            ref_filename2 = run+'_duet2_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_reference.fits'
            ref_hdu_2.writeto(path2+'/'+ref_filename2, overwrite=True)

            # Make source images:
            for srcmag in srcmag_arr:
                src_hdu_1, src_hdu_2 = run_sim(duet=duet, bkg1=bgd_band1, bkg2=bgd_band2,
                                    stack=stack, srcmag=srcmag, nsrc=nsrc, gal=True, gal_params=gal_params, exposure=exposure, frame=frame)
                # Update headers:
                src_hdu_1 = update_header(src_hdu_1, im_type='source', zodi=zodi, gal=gal,
                                    band='DUET1', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
                src_hdu_2 = update_header(src_hdu_2, im_type='source', zodi=zodi, gal=gal,
                                    band='DUET2', srcmag=srcmag, nframes=nsrc, exptime=exposure.value)
                
                # Write files:
                filename1 = run+'_duet1_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_stack-'+str(stack)+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
                src_hdu_1.writeto(path1+'/'+filename1, overwrite=True)
                
                filename2 = run+'_duet2_'+gal+'_'+str(sfb)+'_zodi-'+zodi+'_stack-'+str(stack)+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
                src_hdu_2.writeto(path2+'/'+filename2, overwrite=True)                
                

def sim_path(**kwargs):
    """
    Set path for imsim, make directories if they don't exist.
    Return

    Parameters
    ----------
    run: run identifyer

    gal: galaxy type (or 'none')

    zodi: low, med or high

    band: duet1 or 2

    """
    import os
    import os.path

    run = kwargs.pop('run')
    gal = kwargs.pop('gal')
    zodi = kwargs.pop('zodi')
    band = kwargs.pop('band')

    # Set path:
    path = 'run_'+run+'/gal_'+gal+'/zodi_'+zodi+'/'+band
    # Set file:

    # Make directories if they don't exist
    if not os.path.isdir(path):
        os.makedirs(path)

    return path

def run_sim_ref(**kwargs):
    """
    Run simulations for reference images with given inputs for both DUET bands

    Parameters
    ----------
    duet: Telescope configuration

    bkg1: background sky rate in band 1
    
    bkg2: background sky rate in band 2

    ref_arr: list of reference image depths

    gal: Boolean, trigger galaxy on or off

    gal_params: galaxy parameters, necessary if gal=True

    exposure: quantity, exposure time in seconds

    frame: image frame

    Returns
    -------
    ref_hdu_1, ref_hdu_2: HDU's with simulated reference images
    """
    # Deal with kwargs:
    duet = kwargs.pop('duet')
    bkg1 = kwargs.pop('bkg1')
    bkg2 = kwargs.pop('bkg2')
    ref_arr = kwargs.pop('ref_arr')
    gal = kwargs.pop('gal')
    gal_params = kwargs.pop('gal_params', None)
    exposure = kwargs.pop('exposure')
    frame = kwargs.pop('frame')

    # Make reference images:
    print('Building reference images...')
    empty_hdu_1 = fits.PrimaryHDU()
    empty_hdu_2 = fits.PrimaryHDU()
    ref_hdu_1 = fits.HDUList([empty_hdu_1])
    ref_hdu_2 = fits.HDUList([empty_hdu_2])
    for nref in ref_arr:
        if gal:
            image1 = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=None,
                        sky_rate=bkg1, n_exp=nref, duet=duet, duet_no=1)
            image2 = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=None,
                        sky_rate=bkg2, n_exp=nref, duet=duet, duet_no=2)
        else:
            image1 = construct_image(frame, exposure, gal_type=None, source=None,
                        sky_rate=bkg1, n_exp=nref, duet=duet, duet_no=1)
            image2 = construct_image(frame, exposure, gal_type=None, source=None,
                        sky_rate=bkg2, n_exp=nref, duet=duet, duet_no=2)

        imhdu_1 = fits.ImageHDU(image1.value)
        imhdu_1.header['NFRAMES'] = (nref, 'Number of frames in reference image')
        imhdu_1.header['BUNIT'] = image1.unit.to_string()
        imhdu_1.header['EXPTIME'] = (nref*exposure.value, 'Total exposure time of reference image (s)')
        ref_hdu_1.append(imhdu_1)

        imhdu_2 = fits.ImageHDU(image2.value)
        imhdu_2.header['NFRAMES'] = (nref, 'Number of frames in reference image')
        imhdu_2.header['BUNIT'] = image2.unit.to_string()
        imhdu_2.header['EXPTIME'] = (nref*exposure.value, 'Total exposure time of reference image (s)')
        ref_hdu_2.append(imhdu_2)

    return ref_hdu_1, ref_hdu_2

def run_sim(**kwargs):
    """
    Run simulations for science images with given inputs for both DUET bands

    Parameters
    ----------
    duet: Telescope configuration

    bkg1: background sky rate in band1
    
    bkg2: background sky rate in band2

    stack: number of stacked exposures

    srcmag: source magnitude

    nsrc: number of sources to simulate at each source mag

    gal: Boolean, trigger galaxy on or off

    gal_params: galaxy parameters, necessary if gal=True

    exposure: quantity, exposure time in seconds

    frame: image frame

    Returns
    -------
    im_hdu_1, im_hdu_2: HDU's with simulated images
    """
    # Deal with kwargs:
    duet = kwargs.pop('duet')
    bkg1 = kwargs.pop('bkg1')
    bkg2 = kwargs.pop('bkg2')
    stack = kwargs.pop('stack')
    srcmag = kwargs.pop('srcmag')
    nsrc = kwargs.pop('nsrc')
    gal = kwargs.pop('gal')
    gal_params = kwargs.pop('gal_params', None)
    exposure = kwargs.pop('exposure')
    frame = kwargs.pop('frame')

    # Make source images:
    print('Building source images...')
    empty_hdu_1 = fits.PrimaryHDU()
    src_hdu_1 = fits.HDUList([empty_hdu_1])
    src_fluence_1 = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, 1, duet=duet))
    
    empty_hdu_2 = fits.PrimaryHDU()
    src_hdu_2 = fits.HDUList([empty_hdu_2])
    src_fluence_2 = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, 2, duet=duet))

    for i in range(nsrc):
        source_loc = np.array([np.random.random(), np.random.random()]) # Same in DUET1 and DUET2
        if gal:
            image1 = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=src_fluence_1,
                    source_loc=source_loc, sky_rate=bkg1, n_exp=stack, duet=duet, duet_no=1)
            image2 = construct_image(frame, exposure, gal_type='custom', gal_params=gal_params, source=src_fluence_2,
                    source_loc=source_loc, sky_rate=bkg2, n_exp=stack, duet=duet, duet_no=2)
        else:
            image1 = construct_image(frame, exposure, gal_type=None, source=src_fluence_1,
                    source_loc=source_loc, sky_rate=bkg1, n_exp=stack, duet=duet, duet_no=1)
            image2 = construct_image(frame, exposure, gal_type=None, source=src_fluence_2,
                    source_loc=source_loc, sky_rate=bkg2, n_exp=stack, duet=duet, duet_no=2)
        
        imhdu_1 = fits.ImageHDU(image1.value)
        imhdu_1.header['SRC_POSX'] = (source_loc[0]*frame[0], 'X-position of source in image (pixels)')
        imhdu_1.header['SRC_POSY'] = (source_loc[1]*frame[1], 'Y-position of source in image (pixels)')
        imhdu_1.header['BUNIT'] = image1.unit.to_string()
        imhdu_1.header['EXPTIME'] = (exposure.value*stack, 'Exposure time (s)')
        src_hdu_1.append(imhdu_1)

        imhdu_2 = fits.ImageHDU(image2.value)
        imhdu_2.header['SRC_POSX'] = (source_loc[0]*frame[0], 'X-position of source in image (pixels)')
        imhdu_2.header['SRC_POSY'] = (source_loc[1]*frame[1], 'Y-position of source in image (pixels)')
        imhdu_2.header['BUNIT'] = image2.unit.to_string()
        imhdu_2.header['EXPTIME'] = (exposure.value*stack, 'Exposure time (s)')
        src_hdu_2.append(imhdu_2)

    return src_hdu_1, src_hdu_2

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

def imsim_srcdetect(run='050719',gal='spiral',zodi='low',band='duet1', nmags=71, sfb=[20,30], stack=1):
    """
    Run background estimation, image differencing and source detection on simulated images

    Assumes that the script is run in the directory that contains the run_... directory tree

    Parameters
    ----------
    run: string (date, as in '050719')
        To track runs

    zodi: 'low', 'med' or 'high', default is low

    band: 'duet1' or 'duet2'

    gal: 'spiral', 'elliptical', 'dwarf', or 'none'

    sfb: [sfb_low, sfb_high], default is [20,30]
        List of lowest and highest surface brightness that have been simulated

    nmags: float, default is 71
        Number of source magnitudes used in image simulations

    stack: int, default is 1
        Number of stacked exposures

    Returns
    -------
    run_gal_zodi_band.fits: fits table with source detection results
    """
    # Get telescope configuration from teldef file:
    with open('run_'+run+'/teldef') as origin:
        for line in origin:
            if 'DUET Telescope State' in line:
                tel = line.split(':')[1].strip('\n').strip()

    # Initialize parameters
    duet = Telescope(config=tel)

    # Set up path
    path = 'run_'+run+'/gal_'+gal+'/zodi_'+zodi+'/'+band+'/'

    # Make galaxy surface brightness array
    if gal != 'none':
        sfb_arr = np.arange(sfb[0],sfb[1]+1.).astype(str)

    if band == 'duet1':
        bandpass = duet.bandpass1
        duet_no = 1
    elif band == 'duet2':
        bandpass = duet.bandpass2
        duet_no = 2

    # Make source magnitude array
    src_arr = np.linspace(20.5 - 0.5*(nmags-1)*0.1, 20.5 + 0.5*(nmags-1)*0.1, num=nmags, endpoint=True) # Currently in steps of 0.1 mag

    # Set up results table
    # columns: galaxy mag, source input mag, source input count rate, distance from galaxy center, reference depth, source detected True/False,
    # if True: retrieved count rate, count rate error; number of false positives
    tab = Table(np.zeros(9), names=('galmag', 'srcmag', 'src-ctrate', 'dist', 'ref_depth', 'detected',
                                                    'ctrate', 'ctrate_err', 'false-pos'), dtype=('f8','f8','f8','f8',
                                                    'i8','b','f8','f8','i8'), meta={'name': gal+' - '+zodi+ 'zodi - '+band})

    print('Finding sources...')
    if gal == 'none':
        reffile = run+'_'+band+'_zodi-'+zodi+'_reference.fits'
        hdu_ref = fits.open(path+reffile)

        for srcmag in src_arr:
            imfile = run+'_'+band+'_zodi-'+zodi+'_stack-'+str(stack)+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
            hdu_im = fits.open(path+imfile)
            # Get input countrate
            src_ctrate = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, duet_no, duet=duet))
            # Run source detection for this set of HDUs:
            tab = run_srcdetect(hdu_ref=hdu_ref, hdu_im=hdu_im, tab=tab, duet=duet, sfb=np.nan, srcmag=srcmag, src_ctrate=src_ctrate)
            hdu_im.close()
        hdu_ref.close()

    else:
        for sfb in sfb_arr:
            print('SFB: '+sfb)
            reffile = run+'_'+band+'_'+gal+'_'+sfb+'_zodi-'+zodi+'_reference.fits'
            hdu_ref = fits.open(path+reffile)

            for srcmag in src_arr:
                imfile = run+'_'+band+'_'+gal+'_'+sfb+'_zodi-'+zodi+'_stack-'+str(stack)+'_src-'+"{:5.2f}".format(srcmag)+'.fits'
                hdu_im = fits.open(path+imfile)
                # Get input countrate
                src_ctrate = duet.fluence_to_rate(duet_abmag_to_fluence(srcmag*u.ABmag, duet_no, duet=duet))
                # Run source detection for this set of HDUs:
                tab = run_srcdetect(hdu_ref=hdu_ref, hdu_im=hdu_im, tab=tab, duet=duet, sfb=float(sfb), srcmag=srcmag, src_ctrate=src_ctrate)
                hdu_im.close()
            hdu_ref.close()
    # Save output table
    print('Writing file')
    tab.remove_row(0)
    tab.write('run'+run+'_gal-'+gal+'_zodi-'+zodi+'_stack-'+str(stack)+'-'+band+'.fits', format='fits', overwrite=True)
    print('Done')

def run_srcdetect(**kwargs):
    """
    Run the background estimation, image differencing and source detection for given set of reference and image HDUs.
    Append the results to the input table, return the table.

    Parameters
    ----------
    hdu_ref: input HDU with reference images

    hdu_im: input HDU with science images

    tab: Result table

    duet: Telescope configuration

    sfb: input galaxy surface brightness (float)

    srcmag: input source magnitude (float)

    src_ctrate: input source count rate (float)

    Returns
    -------
    tab: table with source detection results
    """
    hdu_ref = kwargs.pop('hdu_ref')
    hdu_im = kwargs.pop('hdu_im')
    tab = kwargs.pop('tab')
    duet = kwargs.pop('duet')
    sfb = kwargs.pop('sfb', np.nan)
    srcmag = kwargs.pop('srcmag')
    src_ctrate = kwargs.pop('src_ctrate')

    import warnings
    warnings.filterwarnings("ignore") # photutils throws a lot of useless warnings when no peaks are found

    # PSF stuff
    oversample = 5
    pixel_size_init = duet.pixel / oversample
    psf_model = duet.psf_model(pixel_size=pixel_size_init, x_size=25, y_size=25)
    psf_os = psf_model.array
    shape = (5, 5, 5, 5)
    psf_array = psf_os.reshape(shape).sum(-1).sum(1)
    psf_fwhm_pix = duet.psf_fwhm / duet.pixel

    for i in range(hdu_ref[0].header['NEXTEND']):
        # Prepare reference image:
        ref_image_rate = hdu_ref[i+1].data / hdu_ref[i+1].header['EXPTIME'] *u.ph / u.s
        ref_bkg, ref_bkg_rms_median = estimate_background(ref_image_rate, method='1D', sigma=2)
        ref_rate_bkgsub = ref_image_rate - ref_bkg
        s_r = np.sqrt(ref_image_rate)
        sr = np.mean(s_r)
        # Get depth of reference image
        ref_depth = hdu_ref[i+1].header['NFRAMES']

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
            star_tbl, bkg_image, threshold = find(diff_image,psf_fwhm_pix.value,method='daophot')
            # Run photometry
            if len(star_tbl) > 0:
                result, resid = run_daophot(diff_image, threshold, star_tbl, duet=duet)
            else:
                detected = False
                ctrate, ctrate_err = np.nan, np.nan
                fp = len(result)
            
            # Define separation from input source and find nearest source:
            if len(result) > 0:
                sep = np.sqrt((result['x'] - hdu_im[j+1].header['SRC_POSX'])**2 + (result['y'] - hdu_im[j+1].header['SRC_POSY'])**2)
                src = np.argmin(sep)
                if sep[src] < 1.5:
                    detected = True                  
                    ctrate, ctrate_err = result[src]['flux_fit'], result[src]['flux_unc']
                    fp = len(result) - 1
                else:
                    detected = False
                    ctrate, ctrate_err = np.nan, np.nan
                    fp = len(result)
            else:
                detected = False
                ctrate, ctrate_err = np.nan, np.nan
                fp = len(result)
            
            tab.add_row([sfb, srcmag, src_ctrate, dist, ref_depth, detected,
                                                       ctrate, ctrate_err, fp])

    return tab