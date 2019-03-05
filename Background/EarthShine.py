# -*- coding: utf-8 -*-
"""
EarthShine.py
Performs calculations to determine the Earth shine background

    Lookup Earth shine background spectrum at 24 deg from Earth's limb.
    (Supports limb angle as optional input).
    esSpecArray = earthShineSpectrum(wavelengthStart, wavelengthStop, \
                                     limbAng = 24, textfile = '')

    Find Earth Shine normalization factor based on angle from limb
    limbFactor = limbAngleFactor(limbAng)

    Determine the Earth shine background level at a target over an orbit in a 
    given wavelength band.
    backArr = earthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                              inclination, precPhase, \
                              increment=5, altitude=600, earthExclusion=30)

    Plot the Earth Shine background level over a complete orbit for a given
    wavelength band.
    fig, ax = plotEarthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                              inclination, precPhase, \
                              increment=5, altitude=600, earthExclusion=30)

Created on Wed Feb 20 08:00:37 2019

@author: ranquist
"""


"""
esSpecArray = earthShineSpectrum(wavelengthStart, wavelengthStop)

Lookup Earth shine background strength at particular wavelength band at
24 deg from Earth's limb.  While Background_Spectrum.csv has 100 Angstrom
resolution, we interpolate to return 1 Angstrom resolution.

Input:
    wavelengthStart - Start of wavelength range in Angstrom (1000 to 5000)
    wavlengthStop   - End of wavelength range in Angstrom (1000 to 5000)
    
    Requires "Background_Spectrum.csv" to be in your working directory.

Optional Input:
    limbAng  - Instead of default 24 deg limb, can use other limb angle.
    textfile - Writes spectrum to a space-delimited text file with the given
               file name.  (Default: '' Does not write a text file)

Output:
    A numpy 2D array of wavelengths (Å) and flux density (photons/s/cm2/str/Å) 
    at 1 Å resolution.
    
Optional Output:
    A space-delimited text file with the name given in textfile.  Has two 
    columns.  The first is the wavelength (Å), the second is the flux density.
    
Method:
    Instructions and lookup table from Hubble.  We convert the units of
    1e-15 ergs/s/cm2/Angstrom to photons/s/cm2/str/Å.
    (http://www.stsci.edu/hst/cos/documents/handbooks/current/ch07.ETC5.html#422838)

@author: Drake Ranquist
"""
def earthShineSpectrum(wavelengthStart, wavelengthStop, limbAng = 24, textfile = ''):
    
    import sys
    import numpy as np
    from scipy.interpolate import interp1d
    import ReadWriteFile as rw
    import ConvertUnits as un
    
    spectrumFile = "Background_Spectrum.csv"
    spectrumUnits = 1e-15  #File lists values in units of 1e-15 ergs cm^-2 s^-1 Angstrom^-1
    solidAngle = 4.91  #File uses COS instrument which covers area of 4.91 arcsec^2
    
    #Check to see if inputs are valid
    if wavelengthStart > wavelengthStop:
        sys.exit("Starting wavelength must be less then ending wavelength.")
    
    if wavelengthStart < 1000:
        sys.exit("Input start wavelength must be greater than 999 Angstrom.")
        
    if wavelengthStop > 5000:
        sys.exit("Input start wavelength must be less than 5001 Angstrom.")
    
    #Read CSV file and organize for use
    rowHead, _, data = rw.readCSV(spectrumFile)
    
    lambdaArr = np.array([float(ii) for ii in rowHead[1:]])
    
    #Remove space, convert dash to minus sign, and make float
    earthSpec = [ii.replace(' ','').replace("–", "-") for ii in data[:,0]]
    earthSpec = np.array([float(ii) for ii in earthSpec])
    
    #Interpolate over spectrum (in log space)
    earthLogInterp = interp1d(lambdaArr, np.log10(earthSpec))
    
    #Make array of wavelengths
    wavelengths = np.arange(wavelengthStart, wavelengthStop+1)
    
    #Find background in ergs/s/cm2/arcsec2/Angstrom
    earthBack = (10**earthLogInterp(wavelengths)) * spectrumUnits/solidAngle
    
    #Convert to photons/s/cm2/str/Å
    earthBack = un.convertHubbleFluxDens(earthBack, wavelengths)
    
    #Adjust by view angle from limb of Earth
    earthBack *= limbAngleFactor(limbAng)
    
    #Create 2D array to return
    ret2DArray = np.column_stack((wavelengths,earthBack))
    
    #Write file if given
    if textfile is not '':
        np.savetxt(textfile, ret2DArray, fmt='%5i %.5e', delimiter=' ')
    
    return ret2DArray



"""
limbFactor = limbAngleFactor(limbAng)
Find Earth Shine normalization factor based on angle from limb

Input:
    limbAng - Angle from limb of Earth (deg) [5 deg - 180 deg].   
    
Output:
    Normalization factor relative to 24 deg.  Limb angles greater than 64 deg
    return 0.

@author: Drake Ranquist
"""    
def limbAngleFactor(limbAng):
    
    import sys
    import ConvertUnits as un

    refMag = 21.178571428571427  #Magnitude of Earth Shine at 24 deg

    #Find which range angle is in and give points for a line
    #(x1,y1) and (x2,y2)
    if limbAng < 5:
        sys.exit("earthShine does not support angles less than 5 degrees.")        
    elif limbAng < 14:
        x1,y1 = 10, 15.5
        x2,y2 = 14, 19
    elif limbAng < 22:
        x1,y1 = 14, 19
        x2,y2 = 22, 21
    elif limbAng < 50:
        x1,y1 = 22, 21
        x2,y2 = 50, 23.5
    elif limbAng < 64:
        x1,y1 = 50, 23.5
        x2,y2 = 64, 25
    else:
        return 0  #Background level is so low that it is negligable

    #Return background Earth shine from proper linear model
    earthShineMag = (y2-y1)/(x2-x1) * (limbAng - x1) + y1
    
    #Return factor difference between earthShineMag and reference magnitude
    return un.magFactor(earthShineMag, refMag)



"""
backArr = earthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                          inclination, precPhase, \
                          increment=5, altitude=600, earthExclusion=30)

Determine the Earth shine background level at a target over an orbit in a 
given wavelength band.

Inputs:
    ra  - Right ascension of target in degrees (0 to 360)
    dec - Declination of target in degrees (-90 to 90)
    wavelengthStart - Start of wavelength range in Angstrom (1195 to 26957)
    wavlengthStop   - End of wavelength range in Angstrom (1195 to 26957)
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator in
                  degrees.    

Optional inputs:
    increment - The step size between each orbital phase angle in degrees.  
                Must be a factor of 360.  (Default: 5)
    altitude  - Altitude of spacecraft from Earth's surface in km. 
                (Default: 600)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)

Output:
    List of Earth shine background flux densities (photons/s/cm2/str/Å) for 
    each orbital phase of given incremental step size.  For times when target
    is behind the Earth or in the Earth Exclusion area, returns 'EE'.
                  
"""
def earthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                    inclination, precPhase, \
                    increment=5, altitude=600, earthExclusion=30):
    
    import numpy as np
    import Orbit
    import CoordinateTransform as ct
    import zodiac
    
    #Create list of orbital phases
    orbPhases = np.arange(0, 360, increment)

    #Initialize output array
    backArr = [None]*len(orbPhases)

    #Solve for background for each orbital location
    for ii in range(len(orbPhases)):
        
        #Get orbit position
        orbPos = Orbit.orbitPosition(orbPhases[ii], precPhase, inclination)
        
        #Determine angle between position and target
        targetAngle = ct.angleBetweenRaDecs(ra, dec, orbPos[0], orbPos[1])
        
        #Determine angle viewable
        viewAngle = Orbit.viewAngle(altitude, earthExclusion)
        
        #Determine background value if in view
        if targetAngle > viewAngle:
            backArr[ii] = 'EE'
        else:
            orbitToLimbAngle = Orbit.angleToLimb(altitude)
            targetToLimbAngle = 90 + orbitToLimbAngle - targetAngle
            spectrum = earthShineSpectrum(wavelengthStart, wavelengthStop, \
                                          targetToLimbAngle)
            backArr[ii] = zodiac.integrateSpectrum(spectrum)

    #Return array of backgrounds
    return backArr        
        
        

"""
fig, ax = plotEarthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                              inclination, precPhase, \
                              increment=5, altitude=600, earthExclusion=30)

Plot the Earth Shine background level over a complete orbit for a given
wavelength band.

Inputs:
    ra  - Right ascension of target in degrees (0 to 360)
    dec - Declination of target in degrees (-90 to 90)
    wavelengthStart - Start of wavelength range in Angstrom (1195 to 26957)
    wavlengthStop   - End of wavelength range in Angstrom (1195 to 26957)
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator in
                  degrees.    

Optional inputs:
    increment - The step size between each orbital phase angle in degrees.  
                Must be a factor of 360.  (Default: 5)
    altitude  - Altitude of spacecraft from Earth's surface in km. 
                (Default: 600)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)

Output:
    Figure and axis objects of plot of Earth Shine background over orbit.

"""
def plotEarthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                        inclination, precPhase, \
                        increment=5, altitude=600, earthExclusion=30):
    
    import matplotlib.pyplot as plt
    import numpy as np
    #import Orbit as orb
    
    #Replacement value for 'EE'
    eeReplacement = 1e6
    
    
    #Set up the orbital phases
    orbPhases = np.arange(0, 365, 5)

    #Get Earth Shine background
    back = earthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                           inclination, precPhase, \
                           increment=5, altitude=600, earthExclusion=30)    
    
    #Replace 'EE' values and add 360 value
    back.append(back[0])
    back = [eeReplacement if val == 'EE' else val for val in back]
    
    #Plot
    fig, ax = plt.subplots()
    ax.plot(orbPhases, back)
    
    ax.set_title("Earth Shine Background Over Orbit")
    ax.set_xlabel("Orbital Phase (deg)")
    ax.set_ylabel("Flux Density (photons/s/cm$^2$/str/Å)")
    ax.set_xlim([0,360])
    ax.set_yscale('log')
    
    return [fig, ax]
        


"""
fig, ax = plotEarthShineYear(ra, dec, wavelengthStart, wavelengthStop, \
                             altitude=600, earthExclusion=30)

Plots the average Earth Shine background level for a series of inclinations.

Inputs:
    ra  - Right ascension of target in degrees (0 to 360)
    dec - Declination of target in degrees (-90 to 90)
    wavelengthStart - Start of wavelength range in Angstrom (1195 to 26957)
    wavlengthStop   - End of wavelength range in Angstrom (1195 to 26957)

Optional Inputs:
    altitude  - Altitude of spacecraft from Earth's surface in km. 
                (Default: 600)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)  
    year - Year to use for plot (Default: 2019)
    
Output:
    Figure and axis objects of plot of Earth Shine background over year.

"""
def plotEarthShineYear(ra, dec, wavelengthStart, wavelengthStop, \
                       altitude=600, earthExclusion=30, year=2019):

    import matplotlib.pyplot as plt
    import numpy as np
    import Orbit as orb
    
    fig, ax = plt.subplots()
    
    dates = orb.getYearList(year)
    #inclinations = np.arange(0, 100, 10)
    sunSyncInc = orb.sunSyncInclination(altitude)
    inclinations = np.array([0,23.5,30,40,66.5,90,sunSyncInc])
    
    #Orbital increment to use
    increment = 1
    
    backAvg = [0] * len(dates)
    
    for inclination in inclinations:
        for ii in range(len(dates)):
            
            precPhase = orb.getPrecPhase(inclination, dates[ii], vernalPhase=90)
            back = earthShineOrbit(ra, dec, wavelengthStart, wavelengthStop, \
                                   inclination, precPhase, \
                                   increment, altitude, earthExclusion)
            #Remove 'EE' values
            backClean = [val for val in back if val != 'EE']
            backAvg[ii] = np.mean(backClean)

        ax.plot(range(len(dates)), backAvg, label="{:.1f}".format(inclination))
    
    
    ax.set_title("Average Earth Shine vs. Inclination")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Average Flux Density (photons/s/cm$^2$/str/Å)")
    ax.set_ylim([0.1,100])
    ax.set_xlim([0,365])
    ax.set_yscale('log')
    
    ax.legend(title='inclination', ncol=3, loc='lower right')
    
    return [fig,ax]
    








