import numpy as np
import scipy.signal as sig

import astropy.io.fits as fits
from astropy import units as ur

# import IMAGE SIMULATION

import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1.)


def py_zogy(N, R, P_N, P_R, S_N, S_R, SN, SR, dx=0.25, dy=0.25):

    '''Python implementation of ZOGY image subtraction algorithm. Copied from https://github.com/cenko/ZOGY.
    Modified so that inputs are arrays rather than external files.
    Assumes images have been aligned, background subtracted, and gain-matched.

    Arguments:
    N: New image
    R: Reference image
    P_N: PSF of New image
    P_R: PSF or Reference image
    S_N: 2D Uncertainty (sigma) of New image
    S_R: 2D Uncertainty (sigma) of Reference image
    SN: Average uncertainty (sigma) of New image
    SR: Average uncertainty (sigma) of Reference image
    dx: Astrometric uncertainty (sigma) in x coordinate
    dy: Astrometric uncertainty (sigma) in y coordinate

    Returns:
    D: Subtracted image
    P_D: PSF of subtracted image
    S_corr: Corrected subtracted image
    '''

    # Load the PSFs into memory
    P_N_small = P_N
    P_R_small = P_R

    # Place PSF at center of image with same size as new / reference
    P_N = np.zeros(N.shape)
    P_R = np.zeros(R.shape)
    idx = [slice(N.shape[0]//2 - P_N_small.shape[0]//2,
                 N.shape[0]//2 + P_N_small.shape[0]//2 + 1),
           slice(N.shape[1]//2 - P_N_small.shape[1]//2,
                 N.shape[1]//2 + P_N_small.shape[1]//2 + 1)]
    P_N[idx] = P_N_small
    P_R[idx] = P_R_small

    # Shift the PSF to the origin so it will not introduce a shift
    P_N = fft.fftshift(P_N)
    P_R = fft.fftshift(P_R)

    # Take all the Fourier Transforms
    N_hat = fft.fft2(N)
    R_hat = fft.fft2(R)

    P_N_hat = fft.fft2(P_N)
    P_R_hat = fft.fft2(P_R)

    # Fourier Transform of Difference Image (Equation 13)
    D_hat_num = (P_R_hat * N_hat - P_N_hat * R_hat) 
    D_hat_den = np.sqrt(SN**2 * np.abs(P_R_hat**2) + SR**2 * np.abs(P_N_hat**2))
    D_hat = D_hat_num / D_hat_den

    # Flux-based zero point (Equation 15)
    FD = 1. / np.sqrt(SN**2 + SR**2)

    # Difference Image
    # TODO: Why is the FD normalization in there?
    D = np.real(fft.ifft2(D_hat)) / FD

    # Fourier Transform of PSF of Subtraction Image (Equation 14)
    P_D_hat = P_R_hat * P_N_hat / FD / D_hat_den

    # PSF of Subtraction Image
    P_D = np.real(fft.ifft2(P_D_hat))
    P_D = fft.ifftshift(P_D)	
    P_D = P_D[idx]

    # Fourier Transform of Score Image (Equation 17)
    S_hat = FD * D_hat * np.conj(P_D_hat)

    # Score Image
    S = np.real(fft.ifft2(S_hat))

    # Now start calculating Scorr matrix (including all noise terms)

    # Start out with source noise
    # Sigma to variance
    V_N = S_N**2
    V_R = S_R**2

    # Fourier Transform of variance images
    V_N_hat = fft.fft2(V_N)
    V_R_hat = fft.fft2(V_R)

    # Equation 28
    kr_hat = np.conj(P_R_hat) * np.abs(P_N_hat**2) / (D_hat_den**2)
    kr = np.real(fft.ifft2(kr_hat))

    # Equation 29
    kn_hat = np.conj(P_N_hat) * np.abs(P_R_hat**2) / (D_hat_den**2)
    kn = np.real(fft.ifft2(kn_hat))

    # Noise in New Image: Equation 26
    V_S_N = np.real(fft.ifft2(V_N_hat * fft.fft2(kn**2)))

    # Noise in Reference Image: Equation 27
    V_S_R = np.real(fft.ifft2(V_R_hat * fft.fft2(kr**2)))

    # Astrometric Noise
    # Equation 31
    # TODO: Check axis (0/1) vs x/y coordinates
    S_N = np.real(fft.ifft2(kn_hat * N_hat))
    dSNdx = S_N - np.roll(S_N, 1, axis=1)
    dSNdy = S_N - np.roll(S_N, 1, axis=0)

    # Equation 30
    V_ast_S_N = dx**2 * dSNdx**2 + dy**2 * dSNdy**2

    # Equation 33
    S_R = np.real(fft.ifft2(kr_hat * R_hat))
    dSRdx = S_R - np.roll(S_R, 1, axis=1)
    dSRdy = S_R - np.roll(S_R, 1, axis=0)

    # Equation 32
    V_ast_S_R = dx**2 * dSRdx**2 + dy**2 * dSRdy**2

    # Calculate Scorr
    S_corr = S / np.sqrt(V_S_N + V_S_R + V_ast_S_N + V_ast_S_R)

    return D, P_D, S_corr



