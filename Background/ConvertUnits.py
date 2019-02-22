# -*- coding: utf-8 -*-
"""
ConvertUnits.py
Convert various units.

    Convert from arcsecond^2 to steradians.
    steradians = arcsec2ToStr(arcsec2)

    Convert ergs to photons.
    photons = ergsToPhotons(ergs, wavelengths)

    Convert Hubble flux density.
    fluxDensArr = convertHubbleFluxDens(hubbleArr, wavelengthArr)

    Calculate the flux factor difference between two magnitudes
    factor = magFactor(mag1,mag2)

Created on Wed Feb 20 08:20:27 2019

@author: ranquist
"""


"""
steradians = arcsec2ToStr(arcsec2)
Convert from arcsecond^2 to steradians.

Inputs:
    arcsec2 - Single value or numpy array to convert in arcsec^2.

Output:
    Single value or numpy array in steradians.

"""
def arcsec2ToStr(arcsec2):
    
    return arcsec2/4.254517e10



"""
photons = ergsToPhotons(ergs, wavelengths)
Convert ergs to photons.

Inputs:
    ergs        - Numpy array of quantities with units of ergs
    wavelengths - Numpy array of wavelengths with units of Angstroms

    If single value given in one or the other, that is used for every value
    of the other array (e.g. array for hubbleArr, but single wavelength will 
    use that wavelength for all values in hubbleArr).
    
Output:
    Numpy array with units of photons.
    
"""
def ergsToPhotons(ergs, wavelengths):

    return ergs*wavelengths/1.98644746E-08



"""
fluxDensArr = convertHubbleFluxDens(hubbleArr, wavelengthArr)
Convert Hubble flux density.
ergs/s/cm2/arcsec2/Angstrom to photons/s/cm2/str/Å

Inputs:
    hubbleArr     - Numpy array of quantities with units of 
                    ergs/s/cm2/arcsec2/Angstrom
    wavelengthArr - Numpy array of wavelengths in Angstrom
    
    If single value given in one or the other, that is used for every value
    of the other array (e.g. array for hubbleArr, but single wavelength will 
    use that wavelength for 
    all values in hubbleArr).

Output:
    Numpy array where all flux densities are photons/s/cm2/str/Å

"""
def convertHubbleFluxDens(hubbleArr, wavelengthArr):
    
    #Convert arcseconds^2 to steradians
    strArr = hubbleArr/arcsec2ToStr(1)
    
    #Convert ergs to photons and return
    return ergsToPhotons(strArr, wavelengthArr)
    



"""
factor = magFactor(mag1,mag2)
Calculate the flux factor difference between two magnitudes

Input:
    mag1 - The first magnitude
    mag2 - The second magnitude
    
Output:
    Flux factor difference (F1/F2)

@author: Drake Ranquist
"""
def magFactor(mag1,mag2):
    
    return 10**(0.4*(mag2-mag1))
    