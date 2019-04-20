# Import stuff
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.signal import convolve2d
from astroduet.config import Telescope
from astropy.convolution import convolve

# def gaussian_psf(fwhm,patch_size,pixel_size):
#     '''
#     Return 2D array of a symmetric Gaussian PSF
#     
#     Required inputs:
#     fwhm = FWHM in arcsec (4 * u.arcsec)
#     patch_size = Axis sizes of returned PSF patch in pixels (15,15)
#     pixel_size = Angular size of pixel (6 * u.arcsec)
#     '''
#     x = np.linspace(-(patch_size[0] // 2), patch_size[0] // 2, patch_size[0])
#     y = np.linspace(-(patch_size[1] // 2), patch_size[1] // 2, patch_size[1])
#     x, y = np.meshgrid(x,y)
#     
#     sigma_pix = fwhm / (2. * np.sqrt(2 * np.log(2)) * pixel_size)
#     psf = np.exp(-(x**2 + y**2) / (2 * sigma_pix**2)) / (2 * np.pi * sigma_pix**2)
# 
#     return psf

# def duet_psf(patch_size,pixel_size):
#     '''
#     Return 2D array of the double-Gaussian estimated DUET PSF
#     
#     Parameters
#     ----------
#     patch_size : list
#         Axis sizes of returned PSF patch in pixels [15,15]
# 
#     pixel_size: float, astropy units
#         Angular size of pixel (6 * u.arcsec)
#  
#     
#     
#     Examples
#     --------
#     >>> from config import Telescope
#     >>> duet = Telescope()
#     >>> patch_size = [15, 15]
#     >>> psf = duet_psf(patch_size, duet.pixel)
#     >>>    
# 
#     """
#     '''
#     
#     assert type(patch_size) is list, 'patch_size input should be a list'
# 
#     x = np.linspace(-(patch_size[0] // 2), patch_size[0] // 2, patch_size[0])
#     y = np.linspace(-(patch_size[1] // 2), patch_size[1] // 2, patch_size[1])
#     x, y = np.meshgrid(x,y)
#     
#     
#     
#     
#     point_drift = 1 * u.arcsec # To be added to the PSF in quadrature
#     point_jitter = 5 * u.arcsec
#     
#     # Gaussian dimensions given are 3.25 and 6.5 microns.
#     # Assuming plate scale = 0.64 arcsec/micron, sigma = 2.08 and 4.26 arcsec
#     # Take plate scale as a parameter once that's possible
#     sigma = [2.08, 4.26] * u.arcsec
#     fwhm = np.sqrt((2. * np.sqrt(2 * np.log(2)) * sigma)**2 + point_drift**2 + point_jitter**2)
#     sigma_pix = fwhm / (2. * np.sqrt(2 * np.log(2)) * pixel_size)
# 
#     gauss1 = np.exp(-(x**2 + y**2) / (2 * sigma_pix[0]**2)) / (2 * np.pi * sigma_pix[0]**2)
#     gauss2 = 0.1 * np.exp(-(x**2 + y**2) / (2 * sigma_pix[1]**2)) / (2 * np.pi * sigma_pix[1]**2)
#     psf = (gauss1 + gauss2) / np.sum(gauss1 + gauss2)
#     
#     return psf

def sim_galaxy(patch_size,pixel_size,gal_type=None,gal_params=None):
    '''
        Return 2D array of a Sersic profile to simulate a galaxy
        
        Required inputs:
        patch_size = Axis sizes of returned galaxy patch in pixels (15,15)
        pixel_size = Angular size of pixel (6 * ur.arcsec)
        
        Optional inputs:
        gal_type = String that loads a pre-built 'average' galaxy or allows custom definition
        gal_params = Dictionary of parameters for Sersic model: ...
    '''
    from astropy.modeling.models import Sersic2D
    from astroduet.bbmag import bb_abmag_fluence
    from astroduet.duet_telescope import load_telescope_parameters
    
    x = np.linspace(-(patch_size[0] // 2), patch_size[0] // 2, patch_size[0])
    y = np.linspace(-(patch_size[1] // 2), patch_size[1] // 2, patch_size[1])
    x, y = np.meshgrid(x,y)
    
    # Takes either a keyword or Sersic profile parameters
    # Typical galaxy parameters based on Bai et al. (2013)
    # Hardcoded for now, to-do: take distance as an input
    if gal_type == 'spiral':
        # A typical spiral galaxy at 100 Mpc
        amplitude = 0.006 # surface count rate at r_eff
        r_eff = 16.5 / pixel_size.value
        n = 1
        theta = 0
        ellip = 0.5
        x_0, y_0 = r_eff, 0
    elif gal_type == 'elliptical':
        # A typical elliptical galaxy at 100 Mpc
        amplitude = 0.02
        r_eff = 12.5 / pixel_size.value
        n = 4
        theta = 0
        ellip = 0.5
        x_0, y_0 = r_eff, 0
    elif gal_type == 'dwarf':
        # A typical dwarf galaxy at 10 Mpc
        amplitude = 0.009
        r_eff = 70 / pixel_size
        r_eff = r_eff.value
        n = 4
        theta = 0
        ellip = 0.5
        x_0, y_0 = r_eff, 0
    elif (gal_type == 'custom') | (gal_type == None):
        # Get args from gal_params, default to spiral values
        amplitude = gal_params.get('amplitude', 0.006)
        r_eff = gal_params.get('r_eff', 16.5 / pixel_size.value)
        n = gal_params.get('n', 1)
        theta = gal_params.get('theta', 0)
        ellip = gal_params.get('ellip', 0.5)
        x_0 = gal_params.get('x_0', 16.5 / pixel_size.value)
        y_0 = gal_params.get('y_0', 0)
    
    mod = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=x_0, y_0=y_0, ellip=ellip, theta=theta)
    gal = mod(x, y)
    
    return gal / u.s

def construct_image(frame,exposure,read_noise,
                    psf='gaussian', duet=None,
                    gal_type=None,gal_params=None,source=None,sky_rate=None,n_exp=1):
    '''
        Return a simulated image with a background galaxy (optional), source (optional), and noise
        
        Required inputs:
        frame = Pixel dimensions of simulated image (30,30)
        exposure = Integration time of the frame (300 * ur.s)
        psf_fwhm = FWHM in arcsec (4 * ur.arcsec)
        read_noise = Read noise in photons / pixel (unitless float)
        
        Optional inputs:
        psf = "gaussian" or "duet", defaults to "gaussian"
        gal_type = Default galaxy string ("spiral"/"elliptical") or "custom" w/ Sersic parameters in gal_params
        gal_params = Dictionary of parameters for Sersic model (see sim_galaxy)
        source = Source photon count rate (100 / ur.s)
        n_exp = Number of exposures to be co-added
    '''
    
    # Load telescope parameters:
    if duet is None:
        duet = Telescope()

    oversample = 6
    pixel_size_init = duet.pixel / oversample
 
    # Make a PSF kernel
    psf_kernel = duet.psf_model(pixel_size = pixel_size_init)
 
    
#     
#     if psf == 'duet':
#         psf = duet_psf((15,15),pixel_size_init)
#     elif psf == 'gaussian':
#         psf = gaussian_psf(psf_fwhm,(15,15),pixel_size_init)
    
    # Initialise an image, oversampled by the oversample parameter to begin with
    im_array = np.zeros(frame * oversample) / u.s

    # Add a galaxy?
    if gal_type is not None:
        # Get a patch with a simulated galaxy on it
        gal = sim_galaxy(frame * oversample,pixel_size_init,gal_type=gal_type,gal_params=gal_params)
        im_array += gal
    
    # Add a source?
    if source is not None:
        # Place source as a delta function in the center of the frame
        im_array[im_array.shape[0] // 2 + 1, im_array.shape[1] // 2 + 1] += source

    # Convolve with the PSF (need to re-apply units here as it's lost in convolution)

    im_psf = convolve(im_array, psf_kernel)*im_array.unit
#    im_psf = convolve2d(im_array,psf,mode='same') / u.s

    # Apply jitter at this stage? Shuffle everything around by a pixel or something
    
    # Bin up the image by oversample parameter to the correct pixel size
    shape = (frame[0], oversample, frame[1], oversample)
    im_binned = im_psf.reshape(shape).sum(-1).sum(1)
    
    # Now add sky background
    if sky_rate is not None:
        # Add sky rate per pixel across the whole image
        im_binned += sky_rate
    
    # Convert to counts
    im_counts = im_binned * exposure
    # Co-add a number of separate exposures
    im_final = np.zeros(frame)
    for i in range(n_exp):
        # Apply Poisson noise and instrument noise
        im_noise = np.random.poisson(im_counts) + np.random.normal(loc=0,scale=read_noise,size=im_counts.shape)
        im_noise = np.floor(im_noise)
        im_noise[im_noise < 0] = 0

        # Add to the co-add
        im_final += im_noise
    
    # Return image 
    return im_final


def find(image,fwhm,method='daophot',background='2D',diag=False):
    '''
        Find all stars above the sky background level using DAOFind-like algorithm
        
        Required inputs:
        image = 2D array of image on which to perform find
        fwhm = FWHM in pixels (1)
        
        Optional inputs:
        method = Either 'daophot' or 'peaks' to select different finding algorithms
        background = '2D' or 'sigclip' to select 2- or 1-D background estimators
    '''
    from photutils import Background2D, MedianBackground
    from photutils.detection import DAOStarFinder, find_peaks
    from astropy.stats import sigma_clipped_stats
    
    # Create a background image
    if background == '2D':
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, (image.shape[0] // 8, image.shape[1] // 8), bkg_estimator=bkg_estimator)
        sky = bkg.background_rms_median
        bkg_image = bkg.background
    elif background == 'sigclip':
        mean, median, sky = sigma_clipped_stats(image, sigma=3.0)
        bkg_image = image
        bkg_image = median
        
    # Look for five-sigma detections
    threshold = 5 * sky
    
    # Find stars
    if method == 'daophot':
        finder = DAOStarFinder(threshold,fwhm)
        star_tbl = finder.find_stars(image)
        star_tbl['x'], star_tbl['y'] = star_tbl['xcentroid'], star_tbl['ycentroid']
    elif method == 'peaks':
        if background =='sigclip':
            star_tbl = find_peaks(image,threshold,box_size=3)
            star_tbl['x'], star_tbl['y'] = star_tbl['x_peak'], star_tbl['y_peak']
        elif background =='2D':
            star_tbl = find_peaks(image-bkg_image,threshold,box_size=3)
            star_tbl['x'], star_tbl['y'] = star_tbl['x_peak'], star_tbl['y_peak']

    if diag:
        print("Sky background rms: {}".format(sky))
        print("Found {} stars".format(len(star_tbl)))
    
    return star_tbl, bkg_image, threshold

def ap_phot(image,star_tbl,read_noise,exposure,r=1.5,r_in=1.5,r_out=3.):
    '''
        Given an image, go do some aperture photometry
    '''
    from astropy.stats import sigma_clipped_stats
    from photutils import aperture_photometry, CircularAperture, CircularAnnulus
    from photutils.utils import calc_total_error

    # Build apertures from star_tbl
    positions = np.transpose([star_tbl['x'],star_tbl['y']])
    apertures = CircularAperture(positions, r=r)
    annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    annulus_masks = annulus_apertures.to_mask(method='center')

    # Get backgrounds in annuli
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(image)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)

    # Set error
    error = calc_total_error(image, read_noise / exposure, exposure)

    # Perform aperture photometry
    result = aperture_photometry(image, apertures, error=error)
    result['annulus_median'] = bkg_median
    result['aper_bkg'] = bkg_median * apertures.area()
    result['aper_sum_bkgsub'] = result['aperture_sum'] - result['aper_bkg']

    # To-do: fold an error on background level into the aperture photometry error

    for col in result.colnames:
            result[col].info.format = '%.8g'  # for consistent table output
    print("Aperture photometry complete")

    return result, apertures, annulus_apertures

def run_daophot(image,threshold,star_tbl,niters=1, duet=None):
    '''
        Given an image and a PSF, go run DAOPhot PSF-fitting algorithm
    '''
    from photutils.psf import DAOPhotPSFPhotometry, IntegratedGaussianPRF

    
    if duet is None:
        duet = Telescope()
    
    fwhm = (duet.psf_fwhm / duet.pixel).to('').value
    
    # Fix star table columns
    star_tbl['x_0'] = star_tbl['x']
    star_tbl['y_0'] = star_tbl['y']
    
    # Define a fittable PSF model
    sigma = fwhm / (2. * np.sqrt(2 * np.log(2)))
    # Simple Gaussian model to fit
    # to-do: add options to define DUET-like PSF, generate PSF from image etc.
    psf_model = IntegratedGaussianPRF(sigma=sigma)

#    psf_model = duet.psf_model()
#    fwhm = (duet.psf_fwhm / duet.pixel).to('').value
#    print(fwhm)
    
    # Initialise a Photometry object
    # This object loops find, fit and subtract
    photometry = DAOPhotPSFPhotometry(3.*fwhm,threshold,fwhm,psf_model,(5,5),
        niters=niters,sigma_radius=5, aperture_radius=2.)

    
    # Problem with _recursive_lookup while fitting (needs latest version of astropy fix to modeling/utils.py)
    result = photometry(image=image, init_guesses=star_tbl)
    residual_image = photometry.get_residual_image()
    print("PSF-fitting complete")

    return result, residual_image
