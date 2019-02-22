# -*- coding: utf-8 -*-
"""
CoordinateTransform.py
Performs different coordinate tranformations.

    Convert Equatorial to Ecliptic Coordinates
    lat, lon = equatorialToEcliptic(ra,dec)

    Convert geocentric ecliptic longitude to heliocentric ecliptic longitude
    lonHe = geoToHelioEclipticLongitude(lon, year, month, day)

Created on Tue Feb 19 13:46:03 2019

@author: ranquist
"""




"""
Convert Equatorial to Ecliptic Coordinates

Input:
    ra - Right Ascension in degrees (0 to 360)
    dec - Declination in degrees (-90 to 90)
    
Output:
    Two element list with Ecliptic latitude and longitude
    [lat,lon]

@author: Drake Ranquist
"""

def equatorialToEcliptic(ra, dec):
    
    import math
    
    #Constants
    obliq = -23.439   #Obliquity of the ecliptic in degrees
    
    #Convert from degrees to radians
    raRad = math.radians(ra)
    #decRad = math.radians(dec)
    obRad = math.radians(obliq)
    
    #Trig functions on angles
    sinRa = math.sin(raRad)
    cosRa = math.cos(raRad)
    sinOb = math.sin(obRad)
    cosOb = math.cos(obRad)

    """
    Convert equatorial ra and dec to ecliptic lat and lon
    """
    #Convert to equatorial x,y,z
    eqCoLatRad = math.radians(90-dec)
    xEq = math.sin(eqCoLatRad) * cosRa
    yEq = math.sin(eqCoLatRad) * sinRa
    zEq = math.cos(eqCoLatRad)
    
    #Convert to ecliptic x,y,z (Rotate by obliquity)
    xEc = xEq
    yEc = yEq*cosOb - zEq*sinOb
    zEc = yEq*sinOb + zEq*cosOb
    
    #Convert to ecliptic latitude and longitude
    lat = 90 - math.degrees(math.acos(zEc))
    lon = math.degrees(math.atan2(yEc,xEc))
    
    return [lat,lon]
    
    


"""
Convert geocentric ecliptic longitude to heliocentric ecliptic longitude
This is dependant on Earth's location in orbit

Input:
    lon - Geocentric ecliptic longitude (0 to 360)
    date - Either a datetime.date object or three element list with
           [year, month, day]
    
Output:
    Heliocentric ecliptic longitude (0 to 180)
    Values between 180 and 360 are mirrored (0 to 180 = 360 to 180)

@author: Drake Ranquist
"""

def geoToHelioEclipticLongitude(lon, inDate):
    
    import sys
    import datetime as dt
    import numpy as np
    
    #Detect input
    if isinstance(inDate, dt.date):
        thisDate = inDate
    elif isinstance(inDate, (list,np.ndarray)):
        if len(inDate) != 3:
            sys.exit("Date array must have three elements.")
        else:
            thisDate = dt.date(inDate[0], inDate[1], inDate[2])
    else:
        sys.exit("Must supply date as datetime.date object or [year,month,day].")
    
    #Constants
    vernalEquinox = dt.date(thisDate.year, 3, 20)  #Vernal equinox for given year
    daysInYear = 365.25
    
    #Calculate days since vernal equinox
    deltaDays = thisDate - vernalEquinox
    
    #Convert time to longitude
    earthLon = 360 * deltaDays.days/daysInYear
    if earthLon < 0: earthLon += 360
    
    #Get helio-ecliptic longitude (0 to 180 mirrors 360 to 180)
    lonHe = abs(lon - earthLon)
    if lonHe > 180: lonHe = 360 - lonHe
    
    return lonHe