import numpy as np
import matplotlib.pyplot as plt

import sys, getopt

def gauss2D(xy, sigma_x, sigma_y, rho):
    ''' 

    Returns a 2D Gaussian centered at (0,0)

    Arguments:
    xy = tuple with x and y coordinates (assumes mu=0)
    sigma_x = standard deviation in x-coordinate
    sigma_y = standard deviation in y-coordinate
    rho = correlation in x-y plane. -1 < rho < 1
    
    '''
    x, y = xy
    return np.exp((np.square(x/sigma_x) + np.square(y/sigma_y) - (2*rho*x*y/(sigma_x*sigma_y)))/(2*(np.square(rho)-1)))/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-np.square(rho)))


def j_0(xy, k):
    ''' 

    Returns the spherical Bessel function j_0 centered at (0,0)

    Arguments:
    xy = tuple with x and y coordinates (assumes mu=0)
    k = scaling factor; rho=kr
    
    '''
    x, y = xy
    rho = np.sqrt(np.square(x) + np.square(y)) * k
    if rho==0:
        return 1
    else:
        return np.sin(rho)/rho


def PSF(f=gauss2D, f_arg = [5, 5, 0], pix_size=2, patch_size = 100):
    ''' 
    
    Calculates a pixelated PSF given telescope parameters and psf function

    Arguments:
    f = PSF shape; default gaussian.
    f_arg = arguments of the psf function f. Default circular gaussian with sigma 5 arcsec.
    pix_size = size of the pixels in arcsec. This should be smaller than the psf width.
    patch_size = How large in arcsec should the pixelated PSF patch be. Default 100 arcsec
    
    '''

    num_pix = int(patch_size/pix_size)
    mu = (num_pix-1)/2

    pix_psf = np.array([[f(((x-mu) * pix_size, (y-mu) * pix_size), *f_arg) for x in range(num_pix)] for y in range(num_pix)])
    return pix_psf


def main(argv):
    '''
    
    When called from the command line, plot and show a PSF

    '''
    pix_size = 2
    f_arg = [5,5,0]
    try:
        opts, args = getopt.getopt(argv,"hp:s:r:",["pixel_size=","sigma=", "rho="])
    except getopt.GetoptError:
        print('usage: construct_psf.py -p <pixel size (arcsec)> -s <1 sigma of PSF (arcsec)> -r <correlation>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: construct_psf.py -p <pixel size (arcsec)> -s <1 sigma of PSF (arcsec)> -r <correlation>')
            sys.exit()
        elif opt in ("-p", "--pixel_size"):
            pix_size = float(arg)
        elif opt in ("-s", "--sigma"):
            f_arg[0] = float(arg)
            f_arg[1] = float(arg)
        elif opt in ("-r", "--rho"):
            f_arg[2] = float(arg)
    plt.imshow(PSF(pix_size=pix_size, f_arg=f_arg))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Demo
    main(sys.argv[1:])

