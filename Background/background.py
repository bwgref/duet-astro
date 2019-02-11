# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:06:36 2019

Calculate background zodiacal light

Input:
    ra - Right Ascension in degrees (0 to 360)
    dec - Declination in degrees (-90 to 90)
    year - (Ex. 2019)
    month - Numerical month (Ex. 2)
    day - (Ex. 5)
    
    Requires "Zodiac_Light_Table.csv" to be in your working directory.
    
Output:
    Background zodiacal light in V mag arcsec^-2.  
    If within 50 degrees of sun, returns 'SE' for within Solar Exclusion angle.
    
Method:
    Instructions and lookup table from Hubble 
    (http://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html)

@author: Drake Ranquist
"""
def zodiacBackground(ra, dec, year, month, day):
    
    import csv
    import numpy as np

    #Constants
    bgCSVLoc = "Zodiac_Light_Table.csv"

    #Convert equatorial ra and dec to ecliptic lattitude and longitude
    lat, lon = equatorialToEcliptic(ra, dec)
    
    #Convert from geocentric to heliocentric longitude
    lonHe = geoToHelioEclipticLongitude(lon, year, month, day)
    
    """
    Extract Zodiacal Light Data from Hubble CSV file
    """
    bgList = []
    with open(bgCSVLoc) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            bgList.append(row)
    
    #Seperate Headings from data
    bgArr = np.array(bgList)
    bgLons = np.array([int(ii) for ii in bgArr[1:,0]])
    bgLats = np.array([int(ii) for ii in bgArr[0,1:]])
    bgArr = bgArr[1:,1:]
    
    
    """
    Select correct background value
    """
    latInd = (np.abs(bgLats - np.abs(lat))).argmin()
    lonInd = (np.abs(bgLons - lonHe)).argmin()
    
    #Return float if float and string if string
    try:
        float(bgArr[lonInd,latInd])
        return float(bgArr[lonInd,latInd])  #V mag arcsec^-2
    except ValueError:
        return bgArr[lonInd,latInd]  #Solar exclusion
    
    




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
    obliq = -23.446   #Obliquity of the ecliptic in degrees
    
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
    year - (Ex. 2019)
    month - Numerical month (Ex. 2)
    day - (Ex. 5)
    
Output:
    Heliocentric ecliptic longitude (0 to 180)
    Values between 180 and 360 are mirrored (0 to 180 = 360 to 180)

@author: Drake Ranquist
"""

def geoToHelioEclipticLongitude(lon, year, month, day):
    
    from datetime import date

    #Constants
    vernalEquinox = date(year, 3, 20)  #Vernal equinox for given year
    daysInYear = 365.25
    
    #Calculate days since vernal equinox
    thisDate = date(year, month, day)
    deltaDays = thisDate - vernalEquinox
    
    #Convert time to longitude
    earthLon = 360 * deltaDays.days/daysInYear
    if earthLon < 0: earthLon += 360
    
    #Get helio-ecliptic longitude (0 to 180 mirrors 360 to 180)
    lonHe = abs(lon - earthLon)
    if lonHe > 180: lonHe = 360 - lonHe
    
    return lonHe
   


"""
Creates a CSV file with zodiacal light background for range of ra and decs
for vernal equinox 2000

Optional Input:
    increment - Size of increment of table in degrees (Default: 5)
    raStart   - Beginning of right ascension range in degrees (Default: 0)
    raEnd     - End of right ascension range in degrees (Default: 355)
    decStart  - Beginning of declination range in degrees (Default: -90)
    decEnd    - End of declination range in degrees (Default: 90)
    file      - Name/location of CSV file to create 
                  (Default: eqZBackground.csv)
    
Output:
    Writes a CSV file with zodiacal light background for range of ra and decs

@author: Drake Ranquist
"""

def writeZBackgroundCSV(increment=5, raStart=0, raEnd=355, \
                        decStart=-90, decEnd=90, file='eqZBackground.csv'):
    
    import csv
    zbg = zodiacBackground
    
    #Generate First Row
    firstRow = ['']
    raRange = range(raStart, raEnd+increment, increment)
    firstRow += [str(ii) for ii in raRange]
    
    #Open file and write first row
    with open(file, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(firstRow)
        
        #Write Subsequent Lines
        for dec in range(decStart, decEnd+increment, increment):
            nextLine = [str(dec)]
            bgValues = [str(zbg(ra,dec,2000,3,20)) for ra in raRange]
            nextLine += bgValues
            writer.writerow(nextLine)

    return None    

"""
Main Function for command line use.  Runs zodiacBackground
"""
if __name__ == "__main__":
    import sys
    print(zodiacBackground(float(sys.argv[1]),float(sys.argv[2]),\
                     int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])))




