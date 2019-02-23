# -*- coding: utf-8 -*-
"""
zodiac.py
Performs calculations to determine the zodiacal light background

    Lookup zodiacal light background at particular wavelength band at 
    ecliptic poles.
        zodSpecArray = zodiacSpectrum(wavelengthStart, wavelengthStop, textfile='')

    Look up the normalization factor for zodiacal light compared with 
    ecliptic poles based on equatorial right ascension and declination of target.
        normFactor = zodiacLocNorm(ra, dec, date)
        min, avg, max = zodiacLocNorm(ra, dec, startDate, endDate)
    
    Look up closest value in "Zodiac_Light_Table.csv" for given heliocentric
    latitude and longitude.
        zodBack = zodiacLookup(lat, lon)
    

Created on Thu Feb 14 15:51:56 2019

@author: ranquist
"""


"""
zodSpecArray = backSpecAtTarget(ra, dec, wavelengthStart, wavelengthStop)

Get average zodiacal light background spectrum from target.

Inputs:
    ra - Right Ascension in degrees (0 to 360)
    dec - Declination in degrees (-90 to 90)
    wavelengthStart - Start of wavelength range in Angstrom (1195 to 26957)
    wavlengthStop   - End of wavelength range in Angstrom (1195 to 26957)
    
    Requires "Zodiac_Light_Table.csv" to be in your working directory. 
    Requires "Solar_Spectrum.csv" to be in your working directory.

Optional Inputs:
    textfile - Writes spectrum to a space-delimited text file with the given
               file name.  (Default: '' Does not write a text file)
    date - Either a datetime.date object or three element list with
           [year, month, day] for date (Earth location relative to Sun.)
           If not given, find average over entire year.
    endDate - Either a datetime.date object or three element list with
           [year, month, day].  If date and endDate both given, use them as
           date range instead of full year.
           
Output:
    A numpy 2D array of wavelengths (Å) and flux density (photons/s/cm2/str/Å) 
    at 1 Å resolution.
    
Optional Output:
    A space-delimited text file with the name given in textfile.  Has two 
    columns.  The first is the wavelength (Å), the second is the flux density.

"""
def backSpecAtTarget(ra, dec, wavelengthStart, wavelengthStop, \
                     textfile='', date=[2019,1,1], endDate=''):
    
    import numpy as np
    
    #Get Spectrum
    spectrum = zodiacSpectrum(wavelengthStart, wavelengthStop)
    
    #Determine date range
    if endDate == '':
        if date == [2019,1,1]:
            endDate = [2019,12,31]
        else:
            endDate = date
    
    #Get normalization factor
    if endDate == date:
        normFactor = zodiacLocNorm(ra, dec, date)
    else:
        _, normFactor, _ = zodiacLocNorm(ra, dec, date, endDate)
    
    #Adjust spectrum by normalization factor
    #Write to file if requested
    if normFactor == 'SE':
        wavelengths = spectrum[:,0]
        seArray = np.full(len(wavelengths), 'SE')
        spectrum = np.column_stack((wavelengths,seArray))
        if textfile is not '':
            np.savetxt(textfile, spectrum, fmt='%5i %2s', delimiter=' ')
    else:
        spectrum[:,1] *= normFactor
        if textfile is not '':
            np.savetxt(textfile, spectrum, fmt='%5i %.5e', delimiter=' ')
    
    return spectrum
    

"""
zodSpecArray = zodiacSpectrum(wavelengthStart, wavelengthStop)

Lookup zodiacal light background strength at particular wavelength band at
the ecliptic poles.

Input:
    wavelengthStart - Start of wavelength range in Angstrom (1195 to 26957)
    wavlengthStop   - End of wavelength range in Angstrom (1195 to 26957)
    
    Requires "Solar_Spectrum.csv" to be in your working directory.

Optional Input:
    textfile - Writes spectrum to a space-delimited text file with the given
               file name.  (Default: '' Does not write a text file)
    
Output:
    A numpy 2D array of wavelengths (Å) and flux density (photons/s/cm2/str/Å) 
    at 1 Å resolution.
    
Optional Output:
    A space-delimited text file with the name given in textfile.  Has two 
    columns.  The first is the wavelength (Å), the second is the flux density.
    
Method:
    Instructions and lookup table from Indian Institute of Astrophysics 
    (https://cads.iiap.res.in/tools/zodiacalCalc/Documentation).  This is 
    normalized so that can get flux density (photons/s/cm2/str/Å) when 
    multiplied by values in the ecliptic direction table.  We assume the 
    ecliptic poles and multiply by 77.

Modules:
    sys
    numpy
    ReadWriteFile

@author: Drake Ranquist
"""
def zodiacSpectrum(wavelengthStart, wavelengthStop, textfile = ''):
    
    import sys
    import numpy as np
    import ReadWriteFile as rw
    
    #Constants/inputs
    spectrumFile = "Solar_Spectrum.csv"
    poleFactor = 77  #Normalization factor to convert to flux density at poles
    
    #Check to see if inputs are valid
    if wavelengthStart > wavelengthStop:
        sys.exit("Starting wavelength must be less then ending wavelength.")
    
    if wavelengthStart < 1195:
        sys.exit("Input start wavelength must be greater than 1194 Angstrom.")
        
    if wavelengthStop > 26957:
        sys.exit("Input start wavelength must be less than 26,958 Angstrom.")
        
    
    #Read CSV file and organize for use
    rowHead, _, data = rw.readCSV(spectrumFile)
    
    #Convert wavelength and spectrum arrays to floats
    lambdaArr = np.array([float(ii) for ii in rowHead[1:]])
    zodBack = np.array([float(ii) for ii in data])
    
    #Select wavelength range
    startInd = (np.abs(lambdaArr - wavelengthStart)).argmin()
    stopInd  = (np.abs(lambdaArr - wavelengthStop)).argmin()
    
    #Create arrays and convert background to flux density
    wavelengths = lambdaArr[startInd:stopInd+1]
    zodiacLight = zodBack[startInd:stopInd+1] * poleFactor
    ret2DArray = np.column_stack((wavelengths,zodiacLight))
    
    #Write file if given
    if textfile is not '':
        np.savetxt(textfile, ret2DArray, fmt='%5i %.5e', delimiter=' ')
    
    
    return ret2DArray





"""
Look up the normalization factor for zodiacal light compared with 
ecliptic poles based on equatorial right ascension and declination of target.

Input:
    ra - Right Ascension in degrees (0 to 360)
    dec - Declination in degrees (-90 to 90)
    date - Either a datetime.date object or three element list with
           [year, month, day].
    
    Requires "Zodiac_Light_Table.csv" to be in your working directory.  

Optional Input:
    endDate - Either a datetime.date object or three element list with
           [year, month, day].  date and endDate give a range of dates to find
           the min, avg, and max zodiacal background.
    
Output:
    Normalization factor that tells how much brighter the zodiacal light
    is in the given direction than at the ecliptic poles.
    
    If date range input selected, ouput is an array with the minimum, average,
    and maximum normalization factors during that time period.
    
    If within 15 degrees of sun, returns 'SE' for within Solar Exclusion angle.
    
Method:
    Instructions and lookup table from Indian Institute of Astrophysics
    https://cads.iiap.res.in/tools/zodiacalCalc/Documentation

Examples:
    Ecliptic Poles at Vernal Equinox
    zodNorm = zodiacLocNorm(270, 66.56, 2019, 3, 20)

    Min, Average, and Max over an entire year at ra and dec of 10 deg
    dateStart = datetime.date(2019, 1, 1)
    dateEnd = datetime.date(2019, 12, 31)
    minNorm, avgNorm, maxNorm = zodiacLocNorm(10, 10, [dateStart, dateEnd])

Modules:
    sys
    numpy
    datetime
    CoordinateTransform

@author: Drake Ranquist
"""
def zodiacLocNorm(ra, dec, inDate, endDate=None):
    
    import sys
    import numpy as np
    import datetime as dt
    import CoordinateTransform as ct
    
    #Constants
    poleFactor = 77  #Value at ecliptic poles

    #Select start date and convert [year,month,day] to datetime.date object
    if isinstance(inDate, dt.date):
        #Single Date Given
        startDate = inDate
    elif all(isinstance(xx, int) for xx in inDate):
        if len(inDate) != 3:
            sys.exit("Date must be [year, month, day].")
        else:
            #Convert [year,day,month] to a single date
            startDate = dt.date(inDate[0], inDate[1], inDate[2])
    else:
        sys.exit("inDate must be a datetime.date object or a list of 3 integers.")

    
    #If endDate not given, create a range of dates
    if endDate is None:
        dateRange = [startDate]
    else:
        #Convert [year,day,month] to datetime.date object
        if isinstance(endDate, (list, np.ndarray)):
            #Check if either [year,day,month] or [date1, date2]
            if all(isinstance(xx, int) for xx in endDate):
                if len(endDate) != 3:
                    sys.exit("End date must be [year, month, day].")
                else:
                   #Convert [year,day,month] to a single date
                   endDate = dt.date(endDate[0], endDate[1], endDate[2])
            else:
               sys.exit("End date must be [year, month, day] or single datetime.date object.")
        
        #Create list of dates between start and end dates inclusive
        if isinstance(endDate, dt.date):
            if endDate > startDate:
                rIt = range((endDate-startDate).days + 1)
                dateRange = [startDate + dt.timedelta(days=delta) for delta in rIt]
            else:
                sys.exit("Start date must come before end date.")
        else:
            sys.exit("End date must be [year, month, day] or single datetime.date object.")
        

    #Transform equatorial ra and dec to ecliptic lattitude and longitude
    lat, lon = ct.equatorialToEcliptic(ra, dec)
    
    #Transform from geocentric to heliocentric longitude for all dates
    lonHe = [ct.geoToHelioEclipticLongitude(lon, date) for date in dateRange]
    
    #Lookup zodiac background from table for all dates
    zodBack = [zodiacLookup(lat, thisLon) for thisLon in lonHe]
    
    #Return single value if only one date given
    if len(zodBack) == 1:
        if zodBack[0] == 'SE':
            return zodBack[0]
        else:
            return zodBack[0]/poleFactor
    #Return [min, avg, max] if range of dates given
    else:
        if 'SE' in zodBack:
            #Set maximum value to SE and remove all instances from list
            maxValue = 'SE'
            zodNoSE = [value for value in zodBack if value != 'SE']
            if len(zodNoSE) == 0:
                return ['SE','SE','SE']
        else:
            maxValue = max(zodBack)
            zodNoSE = zodBack
        
        minValue = min(zodNoSE)/poleFactor
        avgValue = np.mean(zodNoSE)/poleFactor
        
        return [minValue, avgValue, maxValue]



"""
Look up closest value in "Zodiac_Light_Table.csv" for given heliocentric
latitude and longitude.

Input:
    lat - Heliocentric ecliptic latitude.
    lon - Heliocentric ecliptic longitude.
    
    Requires "Zodiac_Light_Table.csv" to be in your working directory.
    
Output:
    Closest value in "Zodiac_Light_Table.csv"
    
    If within 15 degrees of sun, returns 'SE' for within Solar Exclusion angle.
    
Method:
    Instructions and lookup table from Indian Institute of Astrophysics
    https://cads.iiap.res.in/tools/zodiacalCalc/Documentation

Modules:
    numpy
    ReadWriteFile

@author: Drake Ranquist
"""
def zodiacLookup(lat, lon):
    
    import numpy as np
    import ReadWriteFile as rw
    
    #Constants
    bgCSVLoc = "Zodiac_Light_Table.csv"
    
    """
    Extract Zodiacal Light Data from Hubble CSV file
    """
    #Get headers and data as string arrays
    rowHead, colHead, bgArr = rw.readCSV(bgCSVLoc)
    
    #Convert headers to int arrays and remove first element
    bgLons = np.array([int(ii) for ii in rowHead[1:]])
    bgLats = np.array([int(ii) for ii in colHead[1:]])
    
    
    """
    Select correct background value
    """
    latInd = (np.abs(bgLats - np.abs(lat))).argmin()
    lonInd = (np.abs(bgLons - lon)).argmin()
    
    #Return float if float and string if string
    #Normalize by value at ecliptic poles
    try:
        zodBack = float(bgArr[lonInd,latInd])
        return zodBack
    except ValueError:
        return bgArr[lonInd,latInd]  #Solar exclusion