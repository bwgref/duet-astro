# -*- coding: utf-8 -*-
"""
Orbit.py
Performs various orbital calculations

    Calculate the fraction of the sky that is visible at given orbital phase.
    fractionVisible = instantSkyVisibility(orbPhase, precPhase, inclination,\
                                           altitude, inDate, useSun, useMoon, \
                                           earthExclusion, sunExclusion, \
                                           moonExclusion)

    Calculate the fraction of the sky that is visible over a full orbit.
    fractionVisible = orbitSkyVisibility(precPhase, inclination, altitude, \
                                         inDate, useSun, useMoon, \
                                         earthExclusion, sunExclusion, \
                                         moonExclusion)

    A list of the cumulative fraction of the sky visible for each date.
    visFracArr = cumulativeVisibility(startDate, endDate, inclination,\
                                      altitude, useSun, useMoon, \
                                      earthExclusion, sunExclusion, moonExclusion)

    Checks to see if target is visible.
    boolean = isVisible(ra, dec, orbPhase, precPhase, inclination, altitude, 
                        inDate, useSun, useMoon, \
                        earthExclusion, sunExclusion, moonExclusion)

    Calculate the time a target is visible over an orbit in minutes.    
    time = viewTime(ra, dec, precPhase, inclination, altitude, earthExclusion)
    
    Calculates the spacecraft position.
    ra, dec = orbitPosition(orbPhase, precPhase, inclination)

    Calculates the direction of the angular momentum vector of orbit.
    ra, dec = angMomDirection(precPhase, inclination, reverse=False)

    Find the angle between the spacecraft zenith to the Earth exclusion angle
    angle = viewAngle(altitude, earthExclusion)

    Calculate the phase of the precession on given date.
    precPhase = getPrecPhase(inclination, inDate, altitude, vernalPhase)

    Calculate the rate of the precession from Earth J2 perturbations.
    precRate = getPrecRate(inclination)  

    Calculate the orbital period of spacecraft.
    orbitPeriod = getOrbitPeriod(altitude)

    Returns Earth's Radius (6378.14 km).
    earthRadius = getEarthRadius()

    Get the inclination of a Sun-synchronous circular orbit.
    inclination = sunSyncInclination(altitude)

    Find the location of the Sun.
    ra, dec = sunLocation(date)
    
    Find the location of the Moon.
    ra, dec = moonLocation(date)
    
    Check if the date is in the proper format & convert to datetime.date object.
    date = checkDate(inDate)
    
    Get a list of all dates in a given year as datetime.date objects.
    yearList = getYearList(year)

    Get a list of all dates between two dates as datetime.date objects.
    yearList = dateRangeList(startDate, endDate)

    Angle from a given altitude to the limb of the Earth.
    limbAngle = angleToLimb(altitude)
    
    Find the solid angle of a cone.
    solidAngle = solidAngleCone(angle)
    
    Find the solid angle of the region between two intersecting cones.
    solidAngle = solidAngleIntersection(coneAng1, coneAng2, sepAng)

Created on Fri Feb 22 15:06:52 2019

@author: ranquist
"""



"""
fractionVisible = instantSkyVisibility(orbPhase, precPhase, inclination,\
                                       altitude, inDate, useSun, useMoon, \
                                       earthExclusion, sunExclusion, \
                                       moonExclusion)

Calculate the fraction of the sky that is visible at given orbital phase.

Inputs:
    orbPhase    - Phase of object in orbit (0 to 360) (right handed about
                  angular momentum vector of orbit)
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator in
                  degrees.
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
    inDate      - Either a datetime.date object or [year,month,day] array.
                  (Default: [2019,3,20])
    useSun      - Boolean specifying if include sun in calculations.
                  (Default: True)
    useMoon     - Boolean specifying if include moon in calculations.
                  (Default: True)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)
    sunExclusion   - Minimum angle spacecraft can point towards Sun.
                     (Default: 45)
    moonExclusion  - Minimum angle spacecraft can point towards Moon.
                     (Default: 30)

Output:
    Fraction of sky that is visible to the spacecraft.
    
Notes:
    Assumes that if view intersects both sun and moon that any intersection
    between the two is also visible.  Not completely accurate, but should
    only be a small percentage.

"""
def instantSkyVisibility(orbPhase, precPhase, inclination,\
                         altitude=600, inDate=[2019,3,20], \
                         useSun=True, useMoon=True, \
                         earthExclusion=30, sunExclusion=45, moonExclusion=30):

    import numpy as np
    import CoordinateTransform as coord
    
    fullSky = 4*np.pi
    
    #Calculate max sky visible
    maxAngle = viewAngle(altitude, earthExclusion)
    maxSkyVis = solidAngleCone(maxAngle)

    #Use this if not using Sun or Moon
    if not (useSun or useMoon):
        return maxSkyVis/fullSky

    #Get spacecraft position
    scPosn = orbitPosition(orbPhase, precPhase, inclination)
    
    #Determine view intersection with Sun
    if useSun:
        sunPosn = sunLocation(inDate)
        sunAng = coord.angleBetweenRaDecs( scPosn[0],  scPosn[1], \
                                          sunPosn[0], sunPosn[1])
        sunIntersect = solidAngleIntersection(maxAngle, sunExclusion, sunAng)
    else:
        sunIntersect = 0
    
    #Determine view intersection with Moon    
    if useMoon:
        moonPosn = moonLocation(inDate)
        moonAng = coord.angleBetweenRaDecs(  scPosn[0],   scPosn[1], \
                                           moonPosn[0], moonPosn[1])
        moonIntersect = solidAngleIntersection(maxAngle, moonExclusion, moonAng)
    else:
        moonIntersect = 0
    
    #Determine if Sun and Moon intersect within view
    totalIntersection = sunIntersect + moonIntersect
    if sunIntersect != 0 and moonIntersect !=0:
        sunMoonAng = coord.angleBetweenRaDecs( sunPosn[0],  sunPosn[1], \
                                              moonPosn[0], moonPosn[1])
        sunMoonIntersect = solidAngleIntersection(sunExclusion, moonExclusion,\
                                                  sunMoonAng)
        #Remove intersection
        if sunMoonIntersect > 0:
            totalIntersection -= sunMoonIntersect
    
    
    #Return fraction of sky that is visible
    return (maxSkyVis - totalIntersection)/fullSky




"""
fractionVisible = orbitSkyVisibility(precPhase, inclination, altitude, \
                                     inDate, useSun, useMoon, \
                                     earthExclusion, sunExclusion, \
                                     moonExclusion)

Calculate the fraction of the sky that is visible over a full orbit.

Inputs:
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator in
                  degrees.

Optional Inputs:
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
    inDate      - Either a datetime.date object or [year,month,day] array.
                  (Default: [2019,3,20])
    useSun      - Boolean specifying if include sun in calculations.
                  (Default: True)
    useMoon     - Boolean specifying if include moon in calculations.
                  (Default: True)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)
    sunExclusion   - Minimum angle spacecraft can point towards Sun.
                     (Default: 45)
    moonExclusion  - Minimum angle spacecraft can point towards Moon.
                     (Default: 30)

Output:
    Fraction of sky that is visible to the spacecraft over an orbit.
    
Notes:
    Assumes that the missing area of orbit does not include the intersection
    of both the sun and the moon.  Not completely accurate, but should
    only be a tiny percentage.

"""
def orbitSkyVisibility(precPhase, inclination,\
                       altitude=600, inDate=[2019,3,20], \
                       useSun=True, useMoon=True, \
                       earthExclusion=30, sunExclusion=45, moonExclusion=30):
    
    import numpy as np
    import CoordinateTransform as coord
    
    fullSky = 4*np.pi
    
    #Calculate max sky visible
    instantAngle = viewAngle(altitude, earthExclusion)
    if instantAngle < 90:
        missingZone = 2*solidAngleCone(90-instantAngle)
    else:
        missingZone = 0
        
    maxSkyVis = fullSky - missingZone
    
    #Return this if not using Sun or Moon
    if not (useSun or useMoon):
        return maxSkyVis/fullSky

    
    #Find area missing from Sun
    if useSun:
        sunArea = solidAngleCone(sunExclusion)
    else:
        sunArea = 0
    
    #Find area missing from Moon
    if useMoon:
        moonArea = solidAngleCone(moonExclusion)
    else:
        moonArea = 0
    
    #Get total missing area
    totalMissing = missingZone + sunArea + moonArea
    
    #Add back intersection between sun or moon with missing Zone
    if instantAngle < 90:
        missDir1 = angMomDirection(precPhase, inclination, reverse=False)
        missDir2 = angMomDirection(precPhase, inclination, reverse=True)
        
        if sunArea > 0:
            sunPosn = sunLocation(inDate)
            sunAng1 = coord.angleBetweenRaDecs(missDir1[0],missDir1[1], \
                                                sunPosn[0], sunPosn[1])
            sunAng2 = coord.angleBetweenRaDecs(missDir2[0],missDir2[1], \
                                                sunPosn[0], sunPosn[1])
            sunIntersect1 = solidAngleIntersection(90-instantAngle, \
                                                   sunExclusion, sunAng1)
            sunIntersect2 = solidAngleIntersection(90-instantAngle, \
                                                   sunExclusion, sunAng2)
            totalMissing -= sunIntersect1 + sunIntersect2
        
        if moonArea > 0:
            moonPosn = moonLocation(inDate)
            moonAng1 = coord.angleBetweenRaDecs(missDir1[0], missDir1[1], \
                                                moonPosn[0], moonPosn[1])
            moonAng2 = coord.angleBetweenRaDecs(missDir2[0], missDir2[1], \
                                                moonPosn[0], moonPosn[1])
            moonIntersect1 = solidAngleIntersection(90-instantAngle, \
                                                    moonExclusion, moonAng1)
            moonIntersect2 = solidAngleIntersection(90-instantAngle, \
                                                    moonExclusion, moonAng2)
            totalMissing -= moonIntersect1 + moonIntersect2
    
    
    #Add back intersection of sun and moon
    if sunArea > 0 and moonArea > 0:
        sunPosn = sunLocation(inDate)
        moonPosn = moonLocation(inDate)
        sunMoonAng = coord.angleBetweenRaDecs( sunPosn[0],  sunPosn[1], \
                                              moonPosn[0], moonPosn[1])
        sunMoonIntersect = solidAngleIntersection(sunExclusion, moonExclusion,\
                                                  sunMoonAng)
        totalMissing -= sunMoonIntersect

    
    #Return fraction of sky visible over full orbit
    return 1 - totalMissing/fullSky




"""
visFracArr = cumulativeVisibility(startDate, endDate, inclination,\
                       altitude, useSun, useMoon, \
                       earthExclusion, sunExclusion, moonExclusion)

For each day after the start date, find the fraction of the sky that has been
available to the spacecraft (cumulatively grows from 0 to 1).

Inputs:
    startDate - Start date as a datetime.date object or an array [year,month,day]
    endDate   - End date as a datetime.date object or an array [year,month,day]
    inclination - Inclination of orbit with respect to Earth's equator in
                  degrees.
                 
Optional Inputs:                
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
    inDate      - Either a datetime.date object or [year,month,day] array.
                  (Default: [2019,3,20])
    useSun      - Boolean specifying if include sun in calculations.
                  (Default: True)
    useMoon     - Boolean specifying if include moon in calculations.
                  (Default: True)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)
    sunExclusion   - Minimum angle spacecraft can point towards Sun.
                     (Default: 45)
    moonExclusion  - Minimum angle spacecraft can point towards Moon.
                     (Default: 30)

Output:
    A numpy array with the cumulative fraction of the sky that was visible up
    to that date for every day between the start and end dates (inclusive).

Note:
    Can only be used if none of the unobservable parts of sky intersect
    (e.g. Sun and Moon don't intersect).

"""
def cumulativeVisibility(startDate, endDate, inclination,\
                       altitude=600, useSun=True, useMoon=True, \
                       earthExclusion=30, sunExclusion=45, moonExclusion=30):
    
    import numpy as np
    import CoordinateTransform as coord
    
    dates = dateRangeList(startDate, endDate)
    viewFrac = np.empty(len(dates))
    fullSky = 4*np.pi
    
    #Nothing seen on beginning of first day
    viewFrac[0] = 0
    if len(dates) == 1:
        return viewFrac
    
    
    #One full orbit seen at beginning of second day
    precPhase = getPrecPhase(inclination, dates[1])
    viewFrac[1] = orbitSkyVisibility(precPhase, inclination, \
                                    altitude, dates[1], useSun, useMoon, \
                                   earthExclusion, sunExclusion, moonExclusion) 
    
    #Stop if range is only two days    
    if len(dates) == 2:
        return viewFrac

    #Find initial directions of objects
    initSun = sunLocation(dates[1])
    initMoon = moonLocation(dates[1])
    initAM  = angMomDirection(precPhase, inclination, reverse=False)
    
    #Check if need to use precession
    instantAngle = viewAngle(altitude, earthExclusion)
    if instantAngle < 90:
        missAng = 90 - instantAngle
        usePrecession = True
    else:
        usePrecession = False

    #Slowly add visible regions for each new date
    for ii in range(2, len(dates)):
        thisDate = dates[ii]
        totalMissing = 0
        
        #Find unobservable region from area near angular momentum vector
        if usePrecession:
            precPhase = getPrecPhase(inclination, thisDate)
            amDir  = angMomDirection(precPhase, inclination)
            amAng = coord.angleBetweenRaDecs(initAM[0], initAM[1], \
                                              amDir[0],  amDir[1])
            amSA = solidAngleIntersection(missAng, missAng, amAng)
            if amSA > 0:
                totalMissing += 2*amSA/fullSky
            else:
                usePrecession = False

        #Find unobservable region behind Sun
        if useSun:
            sunDir = sunLocation(thisDate)
            sunAng = coord.angleBetweenRaDecs(initSun[0], initSun[1], \
                                               sunDir[0],  sunDir[1])
            sunSA = solidAngleIntersection(sunExclusion, sunExclusion, sunAng)
            if sunSA > 0:
                totalMissing += sunSA/fullSky
            else:
                useSun = False
                
        #Find unobservable region behind Moon
        if useMoon:
            moonDir = moonLocation(thisDate)
            moonAng = coord.angleBetweenRaDecs(initMoon[0], initMoon[1], \
                                                moonDir[0],  moonDir[1])
            moonSA = solidAngleIntersection(moonExclusion, moonExclusion, moonAng)
            if moonSA > 0:
                totalMissing += moonSA/fullSky
            else:
                useMoon = False

        #Find the fraction of sky seen up to this date
        viewFrac[ii] = 1 - totalMissing

    
    #Return the list of cumulative view fractions
    return viewFrac




"""
boolean = isVisible(ra, dec, orbPhase, precPhase, inclination, altitude, 
                    inDate, useSun, useMoon, \
                    earthExclusion, sunExclusion, moonExclusion)

Checks to see if target is visible.

Inputs:
    ra  - Right ascension of target in degrees (0 to 360)
    dec - Declination of target in degrees (-90 to 90)
    orbPhase    - Phase of spacecraft in orbit (0 to 360) (right handed about
                  angular momentum vector of orbit)
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator in
                  degrees.    

Optional inputs:
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
    inDate      - Either a datetime.date object or [year,month,day] array.
                  (Default: [2019,3,20])
    useSun      - Boolean specifying if include sun in calculations.
                  (Default: True)
    useMoon     - Boolean specifying if include moon in calculations.
                  (Default: True)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)
    sunExclusion   - Minimum angle spacecraft can point towards Sun.
                     (Default: 45)
    moonExclusion  - Minimum angle spacecraft can point towards Moon.
                     (Default: 30)   

Output:
    True if object is visible and False if not.
    
"""
def isVisible(ra, dec, orbPhase, precPhase, inclination, \
              altitude=600, inDate=[2019,3,20], useSun=True, useMoon=True, \
              earthExclusion=30, sunExclusion=45, moonExclusion=30):
    
    import CoordinateTransform as coord
    
    #Get the spacecraft location
    scLoc = orbitPosition(orbPhase, precPhase, inclination)

    #Check if behind or too close to Earth
    angleViewable = viewAngle(altitude, earthExclusion)
    angleFromSpacecraft = coord.angleBetweenRaDecs(ra, dec, scLoc[0], scLoc[1])
    if angleFromSpacecraft > angleViewable:
        return False    
    
    #Check if too close to Sun
    if useSun:
        sunLoc = sunLocation(inDate)
        sunAng = coord.angleBetweenRaDecs(ra, dec, sunLoc[0], sunLoc[1])
        if sunAng <= sunExclusion:
            return False
    
    #Check if too close to Moon
    if useMoon:
        moonLoc = moonLocation(inDate)
        moonAng = coord.angleBetweenRaDecs(ra, dec, moonLoc[0], moonLoc[1])
        if moonAng <= moonExclusion:
            return False
    
    
    #Passed all checks
    return True



"""
time = viewTime(ra, dec, precPhase, inclination, altitude, earthExclusion)

Calculate the time a target is visible over an orbit in minutes.

Inputs:
    ra  - Right ascension of target in degrees (0 to 360)
    dec - Declination of target in degrees (-90 to 90)
    orbPhase    - Phase of spacecraft in orbit (0 to 360) (right handed about
                  angular momentum vector of orbit)
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator in
                  degrees.    

Optional inputs:
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)

Output:
    Time target is visible in minutes (i.e. time not blocked by Earth).

Note:
    Does not check to see if Sun or Moon blocks target.
    
"""
def viewTime(ra, dec, precPhase, inclination, altitude=600, earthExclusion=30):
    
    import numpy as np
    import CoordinateTransform as coord
    
    #Find the period of the orbit
    orbitPeriod = getOrbitPeriod(altitude)
    
    #Find the direction of the angular momentum vector of orbit
    angMomDir = angMomDirection(precPhase, inclination)
    
    #Find angle between target and angular momentum vector
    targAMAng = coord.angleBetweenRaDecs(ra, dec, angMomDir[0], angMomDir[1])
    
    #Find the angle between target and orbital path
    if targAMAng <= 90:
        targetAngle = 90 - targAMAng
    else:
        targetAngle = targAMAng - 90
    
    #Find the angle spacecraft can view
    viewAng = viewAngle(altitude, earthExclusion)
    
    #Check if target is ever in view
    if targetAngle > viewAng:
        return 0
    
    #Find fraction of orbit target is visible
    tAngRad = np.deg2rad(targetAngle)
    vAngRad = np.deg2rad(viewAng)
    visAngRad = np.arccos( np.cos(vAngRad) / np.cos(tAngRad) )
    viewFrac = visAngRad/np.pi
    
    #Return view time of target
    return orbitPeriod*viewFrac


"""
ra, dec = orbitPosition(orbPhase, precPhase, inclination)

Calculates the orbital position (right ascension and declination) given 
the orbital phase, precessional phase, and inclination.  Assumes that 
(ra,dec) = (0,0) for (orbPhase, precPhase) = (0,0) for all inclinations.
Also assumes circular orbits.

Inputs:
    orbPhase    - Phase of object in orbit (0 to 360) (right handed about
                  angular momentum vector of orbit)
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator

Outputs:
    Two element numpy array with the right ascension and declination of 
    object in degrees

"""
def orbitPosition(orbPhase, precPhase, inclination):

    import numpy as np
    import CoordinateTransform as coord

    #Convert from orbital frame to celestial frame
    #Orbital frame:
    #X is from Earth Center to spacecraft, Z is orbital angular momentum
    #vector, Y completes set
    #Celestial frame:
    #X towards Sun, Z along Earth's north Pole, Y completes right handed set
    
    #Spacecraft location in spacecraft frame
    startVector = np.array([1, 0, 0])
    
    #First remove orbital phase
    orbRot = coord.rotationMatrix3D(orbPhase, 'Z')

    #Next remove inclination
    incRot = coord.rotationMatrix3D(inclination, 'X')
    
    #Last, remove precessional phase
    precRot = coord.rotationMatrix3D(precPhase, 'Z')
    
    #Multiply matrices to do full rotation into celestial frame
    endVector = np.squeeze(np.array(precRot @ incRot @ orbRot @ startVector))

    
    #Now convert from cartesian to ra and dec
    ra = np.rad2deg(np.arctan2(endVector[1], endVector[0]))
    dec = 90 - np.rad2deg(np.arccos(endVector[2]))

    return np.array([ra, dec])




"""
ra, dec = angMomDirection(precPhase, inclination, reverse=False)

Calculates the direction (right ascension and declination) of the angular 
momentum vector of orbit.  At a precPhase of 0, the right ascension of the
ascending node is 0.

Inputs:
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node)
    inclination - Inclination of orbit with respect to Earth's equator

Optional Input:
    reverse - Gives the opposite direction of angular momentum vector.
              (Default: False)

Outputs:
    Two element numpy array with the right ascension and declination of 
    angular momentum vector of orbit in degrees

"""
def angMomDirection(precPhase, inclination, reverse=False):
    
    import numpy as np
    
    #Calculate the right ascension
    ra = precPhase + 270
    if ra >= 360: ra -= 360
    
    #Calculate the declination
    dec = 90 - inclination
    
    #Find opposite of angular momentum vector if reverse selected
    if reverse:
        dec = -dec
        
        if ra > 180:
            ra -= 180
        else:
            ra += 180


    return np.array([ra,dec])


"""
angle = viewAngle(altitude, earthExclusion)

Find the angle between the spacecraft zenith to the Earth exclusion angle
(i.e. the angle that corresponds to how much sky can be viewed).

Inputs:
    altitude - Altitude of spacecraft from Earth's surface in km. (Default: 600)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)

Output:
    Angle between spacecraft zenith and closest it can point towards Earth.
    Angle in degrees.

"""
def viewAngle(altitude = 600, earthExclusion = 30):
    
    return 90 + angleToLimb(altitude) - earthExclusion




"""
precPhase = getPrecPhase(inclination, inDate)

Calculate the phase of the precession from Earth J2 perturbations.  Assumes
that the right ascension of the ascending node points towards the vernal
equinox on March 20, 2019.

Input:
    inclination - Inclination of orbit with respect to Earth's equator in 
                  degrees.
    inDate      - Either a datetime.date object or [year,month,day] array.

Optional Input:
    altitude - Altitude of spacecraft from Earth's surface in km. (Default: 600)
    vernalPhase - Phase of precession at vernal equinox (Default: 0)

"""
def getPrecPhase(inclination, inDate, altitude=600, vernalPhase=0):

    import datetime as dt
    
    vernalEq = dt.date(2019,3,20)
    thisDate = checkDate(inDate)
    
    daysDifferent = (thisDate - vernalEq).days
    
    precRate = getPrecRate(inclination)
    totalPrec = precRate * daysDifferent + vernalPhase
    
    #Return the modulo of the total precession
    return totalPrec % 360



"""
precRate = getPrecRate(inclination)

Calculate the rate of the precession from Earth J2 perturbations.  

Input:
    inclination - Inclination of orbit with respect to Earth's equator in 
                  degrees.

Optional Input:
    altitude - Altitude of spacecraft from Earth's surface in km. (Default: 600)
    
Output:
    Precessional rate in degrees/day.

"""
def getPrecRate(inclination, altitude=600):

    import numpy as np
    
    #Calculate needed parameters
    earthRadius = getEarthRadius()
    orbitRadius = earthRadius + altitude
    J2 = 1.08262668e-3
    orbitPeriod = getOrbitPeriod(altitude) * 60 #seconds
    orbitFreq = 2*np.pi/orbitPeriod
    incRad = np.deg2rad(inclination)
    cosi = np.cos(incRad)
    
    #Calculate precession rate in rad/s
    precRate = -1.5 * (earthRadius/orbitRadius)**2 * J2 * orbitFreq * cosi
    
    #Calculate precession rate in degrees/day
    return np.rad2deg(precRate) * 3600 * 24


"""
orbitPeriod = getOrbitPeriod(altitude)

Calculate the orbital period of spacecraft in a circular orbit with a given
altitude.

Input:
    altitude - Altitude of spacecraft from Earth's surface in km.
    
Output:
    Orbital period in minutes.

"""
def getOrbitPeriod(altitude):
    
    earthRadius = getEarthRadius()
    
    return 1.658766e-4 * (altitude + earthRadius)**1.5



"""
earthRadius = getEarthRadius()

Returns Earth's Radius (6378.14 km).
"""
def getEarthRadius():
    
    return 6378.14



"""
inclination = sunSyncInclination(altitude)

Get the inclination of a Sun-synchronous circular orbit.

Inputs:
    altitude - Altitude of spacecraft from Earth's surface in km.
    
Output:
    Inclination, in degrees, of the Sun-synchronous circular orbit.

"""
def sunSyncInclination(altitude):
    
    import numpy as np
    
    #Get the Earth radius
    earthRadius = getEarthRadius()

    #Normalization factor using circular orbit where Sun-synchronous
    #orbits are no longer possible
    sunSyncMax = 12352  #km
    
    #Find cosine of inclination
    cosi = -( (earthRadius + altitude) / sunSyncMax )**3.5
    
    #Return inclination
    return np.rad2deg( np.arccos(cosi) )



"""
ra, dec = sunLocation(date)

Find the location of the Sun in right ascension and declination.

Inputs:
    inDate - Either a datetime.date object or [year,month,day] array.
        
Outputs:
    Two element numpy array with the right ascension and declination of 
    Sun in degrees

"""
def sunLocation(inDate):
    
    import numpy as np
    import ephem

    thisDate = checkDate(inDate)

    sun = ephem.Sun()
    sun.compute(thisDate)

    ra = np.rad2deg(sun.ra)
    dec = np.rad2deg(sun.dec)
    
    return np.array([ra,dec])




"""
ra, dec = moonLocation(date)

Find the location of the Moon in right ascension and declination.

Inputs:
    inDate - Either a datetime.date object or [year,month,day] array.
        
Outputs:
    Two element numpy array with the right ascension and declination of 
    Moon in degrees

"""
def moonLocation(inDate):
    
    import numpy as np
    import ephem

    thisDate = checkDate(inDate)

    moon = ephem.Moon()
    moon.compute(thisDate)

    ra = np.rad2deg(moon.ra)
    dec = np.rad2deg(moon.dec)
    
    return np.array([ra,dec])




"""
date = checkDate(inDate)

Check if the date is in the proper format (either a datetime.date object
or array [year, month, day]).

Input:
    inDate - Either a datetime.date object or [year,month,day] array.
    
Output:
    A datetime.date object if checks pass.

"""
def checkDate(inDate):
    
    import sys
    import datetime as dt
    import numpy as np
    
    if isinstance(inDate, dt.date):
        thisDate = inDate
    elif isinstance(inDate, (list,np.ndarray)):
        if len(inDate) != 3:
            sys.exit("Date array must have three elements.")
        else:
            thisDate = dt.date(inDate[0], inDate[1], inDate[2])
    else:
        sys.exit("Must supply date as datetime.date object or [year,month,day].")
    
    return thisDate



"""
yearList = getYearList(year)

Get a list of all dates in a given year as datetime.date objects.

Input:
    year - Year to get all dates from
    
Output:
    A list of days in year

"""
def getYearList(year):
    
    return dateRangeList([year,1,1], [year,12,31])



"""
yearList = dateRangeList(startDate, endDate)

Get a list of all dates between two dates as datetime.date objects.

Input:
    startDate - Start date as a datetime.date object or an array [year,month,day]
    endDate - End date as a datetime.date object or an array [year,month,day]
    
Output:
    A list of all days between the two dates as datetime.date objects 
    (inclusive).

"""
def dateRangeList(startDate, endDate):
    
    import sys
    import pandas as pd
    import datetime as dt
    
    if endDate < startDate:
        sys.exit("Start date must be before end date!")
    
    startDateFix = checkDate(startDate)
    endDateFix = checkDate(endDate)

    pdDateList = pd.date_range(startDateFix, endDateFix)
    datetimeList = pdDateList.to_pydatetime()

    return [thisDateTime.date() for thisDateTime in datetimeList]



"""
limbAngle = angleToLimb(altitude)

Angle from a given altitude to the limb of the Earth.

Input:
    altitude - Spacecraft altitude from Earth surface in km.
    
Output:
    For a circular orbit, angle from a line tangent to the orbit to the 
    Earth's limb in degrees.

"""
def angleToLimb(altitude):
    
    import numpy as np
    
    earthRadius = getEarthRadius()  #km
    
    return np.rad2deg(np.arccos( earthRadius/(earthRadius+altitude) ))  





"""
solidAngle = solidAngleCone(angle)

Find the solid angle of a cone.

Input:
    angle - Half the angle of the apex of cone (e.g. zenith to edge of view)
            in degrees.
    
Output:
    Solid angle in steradians of cone

"""
def solidAngleCone(angle):
    
    import numpy as np

    return 2*np.pi*( 1 - np.cos(np.deg2rad(angle)) )




"""
solidAngle = solidAngleIntersection(coneAng1, coneAng2, sepAng)

Find the solid angle of the region between two intersecting cones.

Inputs:
    coneAng1 - Half the angle of the apex of first cone in degrees (0 to 90). 
    coneAng2 - Half the angle of the apex of second cone in degrees (0 to 90). 
    sepAng   - Angle between centers of the two cones in degrees (0 to 180).

Output:
    Solid angle in steradians of the intersecting regions of two cones.

Method:
    See https://arxiv.org/ftp/arxiv/papers/1205/1205.1396.pdf.

"""
def solidAngleIntersection(coneAng1, coneAng2, sepAng):
    
    import sys
    import numpy as np
    
    #Check if they intersect
    if sepAng >= coneAng1 + coneAng2:
        return 0  #They don't intersect
    
    #Check if one is inside the other
    if coneAng1 >= coneAng2:
        largeConeAng = coneAng1
        smallConeAng = coneAng2
    else:
        largeConeAng = coneAng2
        smallConeAng = coneAng1

    if largeConeAng >= sepAng + smallConeAng:
        return solidAngleCone(smallConeAng)  #Smaller cone inside larger
    
    #Check if cone angles in right range
    if coneAng1 < 0 or coneAng1 > 90:
        sys.exit("Cone angle must be between 0 and 90 degrees.")
    if coneAng2 < 0 or coneAng2 > 90:
        sys.exit("Cone angle must be between 0 and 90 degrees.")
    if sepAng < 0 or sepAng > 180:
        sys.exit("Seperation angle must be between 0 and 180 degrees.")
    
    #Cones intersect.  Set up rest.
    angRad1 = np.deg2rad(coneAng1)
    angRad2 = np.deg2rad(coneAng2)
    sepAngRad = np.deg2rad(sepAng)
    
    #Trig calculations
    cos1 = np.cos(angRad1)
    cos2 = np.cos(angRad2)
    sinAlpha = np.sin(sepAngRad)
    cosAlpha = np.cos(sepAngRad)
    
    gam1 = np.arctan2(cos2 - cosAlpha*cos1, sinAlpha*cos1)
    gam2 = np.arctan2(cos1 - cosAlpha*cos2, sinAlpha*cos2)
    
    beta1 = np.arccos(np.sin(gam1) / np.sin(angRad1))
    beta2 = np.arccos(np.sin(gam2) / np.sin(angRad2))
    
    phi1 = np.arccos(np.tan(gam1) / np.tan(angRad1))
    phi2 = np.arccos(np.tan(gam2) / np.tan(angRad2))
    
    sigma1 = 2*(beta1 - phi1 * cos1)
    sigma2 = 2*(beta2 - phi2 * cos2)
    
    return sigma1 + sigma2



