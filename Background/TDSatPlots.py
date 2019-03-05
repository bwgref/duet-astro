# -*- coding: utf-8 -*-
"""
TDSatPlots.py
Code for various plots created for TD-Sat.

    Plot the fraction of sky visible to TD-Sat over an orbit for various
    inclinations.
    fig, ax = plotInstantaneousView(precPhase, altitude, inDate, \
                                   useSun, useMoon, \
                                   earthExclusion, sunExclusion, moonExclusion)
    
    Plot the fraction of sky TD-Sat sweeps through during an entire orbit for
    every day of the year and for various inclinations.
    fig, ax = plotOrbitView(year, altitude, useSun, useMoon, \
                           earthExclusion, sunExclusion, moonExclusion)
    
    For each day after the start date, plots the fraction of the sky that has 
    been available to the spacecraft for a variety of inclinations.
    fig, ax = plotCumulativeView(startDate, endDate, altitude, useSun, useMoon,\
                                earthExclusion, sunExclusion, moonExclusion)

    Plot when the north ecliptic pole is visible during an orbit for various
    inclinations.
    fig, ax = plotIsVisEclPole(precPhase, altitude, inDate, useSun, useMoon, \
                          earthExclusion, sunExclusion, moonExclusion)

    Plot the time that the north ecliptic pole is visible over an orbit through
    all of its precessional phases for various inclinations.
    figure = plotEclPoleTimePrec(altitude, earthExclusion)

    Plot the time that the north ecliptic pole is visible over an orbit for every
    day of year for various inclinations.
    figure = plotEclPoleTime(year, altitude, earthExclusion)


Created on Mon Feb 25 16:17:41 2019

@author: ranquist
"""


import matplotlib.pyplot as plt
import numpy as np



""""
figure = plotInstantaneousView(precPhase, altitude, inDate, useSun, useMoon, \
                               earthExclusion, sunExclusion, moonExclusion)

Plot the fraction of sky visible to TD-Sat over an orbit for various
inclinations.

Optional Inputs:
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node) (Default: 0)
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
    Figure and axis objects of plot of fraction of sky that is visible to TD-Sat.
    
"""
def plotInstantaneousView(precPhase=0, altitude=600, inDate=[2019,3,20], \
                          useSun=True, useMoon=True, \
                          earthExclusion=30, sunExclusion=45, moonExclusion=30):
    
    import Orbit as orb
    
    fig, ax = plt.subplots()
    
    orbPhases = np.arange(0, 360)
    #inclinations = np.arange(0, 100, 10)
    sunSyncInc = orb.sunSyncInclination(altitude)
    inclinations = np.array([0,23.5,30,40,66.5,90,sunSyncInc])
    
    for inclination in inclinations:
        intView = [orb.instantSkyVisibility(orbPhase, precPhase, inclination, \
                       altitude, inDate, useSun, useMoon, \
                       earthExclusion, sunExclusion, moonExclusion) \
                   for orbPhase in orbPhases]

        ax.plot(orbPhases, intView, label="{:.1f}".format(inclination))
    
    
    ax.set_title("Instantaneous View vs Inclination (with Sun and Moon)")
    ax.set_xlabel("Orbital Phase (deg)")
    ax.set_ylabel("Instantaneous Fraction of Sky Visible")
    ax.set_ylim([0,0.5])
    ax.set_xlim([0,360])
    ax.legend(title='inclination', loc='lower center')
    
    return [fig, ax]




""""
figure = plotOrbitView(year, altitude, useSun, useMoon, \
                       earthExclusion, sunExclusion, moonExclusion)

Plot the fraction of sky TD-Sat sweeps through during an entire orbit for
every day of the year and for various inclinations.

Optional Inputs:
    year - Year to use for plot (Default: 2019)
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
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
    Figure and axis objects of plot of fraction of sky that is visible to TD-Sat.
    
"""
def plotOrbitView(year=2019, altitude=600, useSun=True, useMoon=True, \
                  earthExclusion=30, sunExclusion=45, moonExclusion=30):
    
    import Orbit as orb
    
    fig, ax = plt.subplots()
    
    dates = orb.getYearList(year)
    #inclinations = np.arange(0, 100, 10)
    sunSyncInc = orb.sunSyncInclination(altitude)
    inclinations = np.array([0,23.5,30,40,66.5,90,sunSyncInc])
    
    orbVis = [0] * len(dates)
    
    for inclination in inclinations:
        for ii in range(len(dates)):
            
            precPhase = orb.getPrecPhase(inclination, dates[ii], vernalPhase=90)
            orbVis[ii] = orb.orbitSkyVisibility(precPhase, inclination, \
                             altitude, dates[ii], useSun, useMoon, \
                             earthExclusion, sunExclusion, moonExclusion) 

        ax.plot(range(len(dates)), orbVis, label="{:.1f}".format(inclination))
    
    
    ax.set_title("View Per Orbit vs Inclination (with Sun and Moon)")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Fraction of Sky Visible Over 1 Orbit on Day")
    ax.set_ylim([0.7,0.9])
    #ax.set_xlim([0,360])
    ax.legend(title='inclination', ncol=3)
    
    return [fig,ax]




""""
figure = plotCumulativeView(startDate, endDate, \
                            altitude, useSun, useMoon, \
                            earthExclusion, sunExclusion, moonExclusion)

For each day after the start date, plots the fraction of the sky that has been
available to the spacecraft (cumulatively grows from 0 to 1).  Does this for
a variety of inclinations.

Inputs:
    startDate - Start date as a datetime.date object or an array [year,month,day]
    endDate   - End date as a datetime.date object or an array [year,month,day]

Optional Inputs:
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
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
    Figure and axis objects of plot of the cumulative fraction of sky that is 
    visible to TD-Sat.
    
"""
def plotCumulativeView(startDate, endDate, \
                       altitude=600, useSun=True, useMoon=True, \
                       earthExclusion=30, sunExclusion=45, moonExclusion=30):
    
    import Orbit as orb
    
    fig, ax = plt.subplots()
    
    #inclinations = np.arange(0, 100, 10)
    sunSyncInc = orb.sunSyncInclination(altitude)
    inclinations = np.array([0,23.5,30,40,66.5,90,sunSyncInc])
    nDays = len(orb.dateRangeList(startDate, endDate))
    
    for inclination in inclinations:
        
        cumVis = orb.cumulativeVisibility(startDate, endDate, inclination, \
                                          altitude, useSun, useMoon, \
                                          earthExclusion, sunExclusion, \
                                          moonExclusion)

        ax.plot(range(nDays), cumVis, label="{:.1f}".format(inclination))
    
    
    ax.set_title("Cumulative View vs Inclination")
    ax.set_xlabel("Days from Vernal Equinox")
    ax.set_ylabel("Cumulative Fraction of Sky Visible")
    ax.set_ylim([0.7,1])
    ax.set_xlim([0,120])
    ax.legend(title='inclination', ncol=3, loc='lower right')
    
    return [fig,ax]






""""
figure = plotIsVisEclPole(precPhase, altitude, inDate, useSun, useMoon, \
                          earthExclusion, sunExclusion, moonExclusion)

Plot when the north ecliptic pole is visible during an orbit for various
inclinations.

Optional Inputs:
    precPhase   - Phase of the precession of the orbit (0 to 360) (right 
                  ascension of acending node) (Default: 0)
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
    Figure and axis objects of plot of fraction of sky that is visible to TD-Sat.
    
"""
def plotIsVisEclPole(precPhase=0, altitude=600, inDate=[2019,3,20], \
                     useSun=True, useMoon=True, \
                     earthExclusion=30, sunExclusion=45, moonExclusion=30):
    
    import Orbit as orb
    
    #Define North Ecliptic Pole location
    nePole = [270, 66.5]
    
    fig, ax = plt.subplots()
    
    orbPhases = np.arange(0, 360)
    
    #Inclinations out of order to put 23.5 line on top
    sunSyncInc = orb.sunSyncInclination(altitude)
    inclinations = np.array([0,23.5,30,40,66.5,90,sunSyncInc])
    #inclinations = np.array([0,23.5,30,40,66.5,90])
    
    for inclination in inclinations:
        isVisible = [orb.isVisible(nePole[0], nePole[1], \
                                   orbPhase, precPhase, inclination, \
                                   altitude, inDate, useSun, useMoon, \
                                   earthExclusion, sunExclusion, moonExclusion) \
                     for orbPhase in orbPhases]

        ax.plot(orbPhases, isVisible, label="{:.1f}".format(inclination))
    
    
    ax.set_title("When is North Ecliptic Pole Visible?")
    ax.set_xlabel("Orbital Phase (deg)")
    ax.set_ylabel("Is Visible? (1=Yes, 0=No)")
    ax.set_xlim([0,360])
    
    handles,labels = ax.get_legend_handles_labels()
    sortL, sortH = (list(x) for x in zip(*sorted(zip(labels, handles))))
    
    ax.legend(sortH, sortL, title='inclination', loc='center right')
    
    
    
    return [fig,ax]





""""
figure = plotEclPoleTimePrec(altitude, earthExclusion)

Plot the time that the north ecliptic pole is visible over an orbit through
all of its precessional phases for various inclinations.

Optional Inputs:
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)

Output:
    Figure and axes objects of plot of time North Ecliptic Pole is visible 
    to TD-Sat.
    
"""
def plotEclPoleTimePrec(altitude=600, earthExclusion=30):
    
    import Orbit as orb
    
    fig, ax = plt.subplots()
    
    #Define North Ecliptic Pole location
    nePole = [270, 66.5]
    
    #Set up x axis
    precPhases = np.arange(0, 361)
    
    #Set up multiple plots
    #inclinations = np.arange(0, 100, 10)
    #inclinations = np.append(inclinations, orb.sunSyncInclination(altitude))
    sunSyncInc = orb.sunSyncInclination(altitude)
    inclinations = np.array([0,23.5,30,40,66.5,90,sunSyncInc])
    
    #Make all plots
    for inclination in inclinations:
        viewTime = [orb.viewTime(nePole[0], nePole[1], precPhase, inclination,\
                                 altitude, earthExclusion) \
                    for precPhase in precPhases]

        ax.plot(precPhases, viewTime, label="{:.1f}".format(inclination))
    
    ax.set_title("Time North Ecliptic Pole Visible Over Full Precession")
    ax.set_xlabel("Precession Phase (deg)")
    ax.set_ylabel("Time Ecliptic Pole Visible (minutes)")
    ax.set_ylim([0,50])
    ax.set_xlim([0,360])
    ax.legend(title='inclination')
    
    return [fig, ax]




""""
figure = plotEclPoleTime(year, altitude, earthExclusion)

Plot the time that the north ecliptic pole is visible over an orbit for every
day of year for various inclinations.

Optional Inputs:
    year - Year to use for plot (Default: 2019)
    altitude    - Altitude of spacecraft from Earth's surface in km. 
                  (Default: 600)
    earthExclusion - Minimum angle spacecraft can point towards Earth.
                     (Default: 30)

Output:
    Figure and axes objects of plot of time North Ecliptic Pole is visible 
    to TD-Sat.
    
"""
def plotEclPoleTime(year=2019, altitude=600, earthExclusion=30):
    
    import Orbit as orb
    
    fig, ax = plt.subplots()
    
    #Define North Ecliptic Pole location
    nePole = [270, 66.5]
    
    dates = orb.getYearList(year)
    #inclinations = np.arange(0, 100, 10)
    sunSyncInc = orb.sunSyncInclination(altitude)
    inclinations = np.array([0,23.5,30,40,66.5,90,sunSyncInc])
    
    timeVis = [0] * len(dates)
    
    for inclination in inclinations:
        for ii in range(len(dates)):
            
            precPhase = orb.getPrecPhase(inclination, dates[ii], vernalPhase=90)
            timeVis[ii] = orb.viewTime(nePole[0], nePole[1], precPhase, \
                                       inclination, altitude, earthExclusion)

        ax.plot(range(len(dates)), timeVis, label="{:.1f}".format(inclination))
    
    
    ax.set_title("Time North Ecliptic Pole Visible Over Year")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Time Ecliptic Pole Visible Over Single Orbit (minutes)")
    ax.set_ylim([0,50])
    ax.set_xlim([1,365])
    ax.legend(title='inclination', ncol=4, loc='lower right')
    
    return [fig,ax]


