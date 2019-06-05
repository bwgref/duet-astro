import os

from astropy import log
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, QTable, join

from astroduet.config import Telescope
from astroduet.bbmag import bb_abmag, bb_abmag_fluence
from .utils import tqdm

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'data')

class Simulations():
    '''
    Container class for the list of simulations and some helper scripts to process them

    Attributes
    ----------
    self.emgw_simulations: string array
        List of EMGW Simulations




    '''


    def __init__(self):

        self.emgw_simulations = ['shock_2.5e10.dat',
            'shock_5e10.dat',
            'shock_1e11.dat',
            'kilonova_0.01.dat',
            'kilonova_0.02.dat',
            'kilonova_0.04.dat']
        self.sne_simulations = ['IIb', 'IIP', 'IIP_big', 'stripped']
        self.sne_rsg_simulations = ['rsg400.dat',
            'rsg450.dat',
            'rsg500.dat',
            'rsg550.dat',
            'rsg600.dat',
            'rsg650.dat',
            'rsg700.dat',
            'rsg750.dat',
            'rsg800.dat',
            'rsg850.dat',
            'rsg900.dat',
            'rsg950.dat',
            'rsg1000.dat']
        self.sne_bsg_simulations = ['bsg20.dat',
            'bsg25.dat',
            'bsg30.dat',
            'bsg35.dat',
            'bsg40.dat',
            'bsg45.dat',
            'bsg50.dat',
            'bsg55.dat',
            'bsg60.dat',
            'bsg65.dat',
            'bsg70.dat',
            'bsg75.dat',
            'bsg80.dat']

    def info(self):
        print('-----')
        print('DUET Simulations:')
        print('-----')
        print('EMGW Simulations Inputs:')


        for emgw in self.emgw_simulations:
            print(emgw)
        print()


        print()

    def parse_emgw(self, diag=False, list_of_simulations=None):
        '''
        Loop over each EMGW GRB shock model and save the outputs

        Optional parameters
        -------------------

        diag: boolean
            Just run one test instead of looping over all for unit tests


        '''

        self.emgw_processed = np.array([])
        if list_of_simulations is None:
            list_of_simulations = self.emgw_simulations
        for ind, shockf in enumerate(list_of_simulations):
            if diag is True:
                if ind > 0:
                    break
            sname, ext = os.path.splitext(shockf)
            print('Parsing and storing: {}'.format(sname))
            outfile = datadir + '/' + sname + '_lightcurve_DUET.fits'
            shock_lc = convert_model(datadir + '/' + shockf, name=sname)
            shock_lc.write(outfile, format='fits', overwrite=True)

        return

    def parse_sne(self, diag=False, list_of_simulations=None):
        '''
        Loop over each SN model and save the outputs

        Optional parameters
        -------------------

        diag: boolean
            Just run one test instead of looping over all for unit tests


        '''

        self.emgw_processed = np.array([])
        if list_of_simulations is None:
            list_of_simulations = self.sne_simulations
        for ind, shockf in enumerate(list_of_simulations):
            if diag is True:
                if ind > 0:
                    break
            sname, ext = os.path.splitext(shockf)
            print('Parsing and storing: {}'.format(sname))
            outfile = datadir+'/'+sname+'_lightcurve_DUET.fits'
            shock_lc = convert_sn_model(datadir + '/' + shockf, name=sname)

            shock_lc.write(outfile, format='fits', overwrite=True)

        return

    def parse_sne_rsg(self, diag=False, list_of_simulations=None):
        '''
        Loop over each RSG SN model and save the outputs

        Optional parameters
        -------------------

        diag: boolean
            Just run one test instead of looping over all for unit tests


        '''

        self.emgw_processed = np.array([])
        if list_of_simulations is None:
            list_of_simulations = self.sne_rsg_simulations
        for ind, shockf in enumerate(list_of_simulations):
            if diag is True:
                if ind > 0:
                    break
            sname, ext = os.path.splitext(shockf)
            print('Parsing and storing: {}'.format(sname))
            outfile = datadir+'/'+sname+'_lightcurve_DUET.fits'
            shock_lc = convert_model(datadir + '/' + shockf, name=sname)

            shock_lc.write(outfile, format='fits', overwrite=True)

        return

    def parse_sne_bsg(self, diag=False, list_of_simulations=None):
        '''
        Loop over each RSG SN model and save the outputs

        Optional parameters
        -------------------

        diag: boolean
            Just run one test instead of looping over all for unit tests


        '''

        self.emgw_processed = np.array([])
        if list_of_simulations is None:
            list_of_simulations = self.sne_bsg_simulations
        for ind, shockf in enumerate(list_of_simulations):
            if diag is True:
                if ind > 0:
                    break
            sname, ext = os.path.splitext(shockf)
            print('Parsing and storing: {}'.format(sname))
            outfile = datadir+'/'+sname+'_lightcurve_DUET.fits'
            shock_lc = convert_model(datadir + '/' + shockf, name=sname)

            shock_lc.write(outfile, format='fits', overwrite=True)

        return


def convert_sn_model(label, name='NoName', duet=None):
    '''
    Reads in the SNe models, converts them to DUET fluences, and
    writes out the resulting models to FITS files.

    Parameters
    ----------

    filename : string
        Path to SN shock file.

    Other parameters
    ----------------

    name : string
        name to use for the model. Default is 'NoName'

    '''
    if duet is None:
        duet = Telescope()

    bandone = duet.bandpass1
    bandtwo = duet.bandpass2
    dist0 = 10*u.pc

    temptable = \
        Table.read(f'{label}_teff.txt', format='ascii', names=['time', 'T'])
    radiustable = \
        Table.read(f'{label}_radius.txt', format='ascii', names=['time', 'R'])
    table = join(temptable, radiustable)
    N = len(table['time'])
    time = table['time'] * u.s

    shock_lc = Table([time,
            np.zeros(len(time))*u.ABmag,
            np.zeros(len(time))*u.ABmag,
            np.zeros(len(time))*u.ph/(u.s*u.cm**2),
            np.zeros(len(time))*u.ph/(u.s*u.cm**2)],
               names=('time', 'mag_D1', 'mag_D2', 'fluence_D1', 'fluence_D2'),
               meta={'name': name + ' at 10 pc',
                      'dist0_pc' : '{}'.format(dist0.to(u.pc).value)})

    bolflux = (table['T'] * u.K) ** 4 * const.sigma_sb.cgs * (
                (table['R'] * u.cm) / dist0.to(u.cm)) ** 2

    temps = table['T'] * u.K

    for k, t, bf in tqdm(list(zip(np.arange(N), temps, bolflux))):
        band1_mag, band2_mag = bb_abmag(bbtemp=t, bolflux = bf,
                        bandone=bandone, bandtwo=bandtwo, val=True)

        band1_fluence, band2_fluence = bb_abmag_fluence(bbtemp=t,
            bolflux=bf)

        shock_lc[k]['mag_D1'] = band1_mag
        shock_lc[k]['mag_D2'] = band2_mag
        shock_lc[k]['fluence_D1'] = band1_fluence.value
        shock_lc[k]['fluence_D2'] = band2_fluence.value

    shock_lc['mag_D1'].unit = None
    shock_lc['mag_D2'].unit = None
    return shock_lc

def convert_model(filename, name='NoName', duet=None):
    '''
    Reads in the EMGW shock breakout models, converts them to DUET fluences, and
    writes out the resulting models to FITS files.

    Parameters
    ----------

    filename : string
        Path to GRB shock file.

    Other parameters
    ----------------

    name : string
        name to use for the model. Default is 'NoName'

    '''

    if duet is None:
        duet = Telescope()

    bandone = duet.bandpass1
    bandtwo = duet.bandpass2
    dist0 = 10*u.pc

    shock_data = np.loadtxt(filename)

    time = (shock_data[:,0]*u.d).to(u.s)
    temps = shock_data[:,2]
    bolflux = 10**shock_data[:,1]

    # Set up outputs
    shock_lc = Table([time,
            np.zeros(len(time))*u.ABmag,
            np.zeros(len(time))*u.ABmag,
            np.zeros(len(time))*u.ph/(u.s*u.cm**2),
            np.zeros(len(time))*u.ph/(u.s*u.cm**2)],
               names=('time', 'mag_D1', 'mag_D2', 'fluence_D1', 'fluence_D2'),
               meta={'name': name + ' at 10 pc',
                      'dist0_pc' : '{}'.format(dist0.to(u.pc).value)})
    N = len(temps)
    for k, t, bf in tqdm(list(zip(np.arange(N), temps, bolflux))):
        t *= u.K
        bf *= (u.erg/u.s) /(4 * np.pi * dist0**2)

        band1_mag, band2_mag = bb_abmag(bbtemp=t, bolflux = bf,
                        bandone=bandone, bandtwo=bandtwo, val=True)

        band1_fluence, band2_fluence = bb_abmag_fluence(bbtemp=t,
            bolflux=bf)

        shock_lc[k]['mag_D1'] = band1_mag
        shock_lc[k]['mag_D2'] = band2_mag
        shock_lc[k]['fluence_D1'] = band1_fluence.value
        shock_lc[k]['fluence_D2'] = band2_fluence.value

    shock_lc['mag_D1'].unit = None
    shock_lc['mag_D2'].unit = None

    return shock_lc


def load_model_fluence(filename, dist=100*u.Mpc):
    '''
    Reads in a FITS version of the model template and scales to the given distance.

    Parameters
    ----------

    filename : string
        Path to model FITS.

    Other parameters
    ----------------

    dist : float
        Distance at which to place the soruce

    Returns
    -------

    Fluence in both DUET bands



    '''

    fitsfile = fits_file(filename)

    model_lc_table = QTable.read(fitsfile)
    model_lc_table['time'] -= model_lc_table['time'][0]

    dist0 = float(model_lc_table.meta['DIST0_PC']) * u.pc

    distscale = (dist0.to(u.Mpc) / dist)**2

    fluence1 = model_lc_table['fluence_D1'] * distscale
    fluence2 = model_lc_table['fluence_D2'] * distscale

    return model_lc_table['time'], fluence1, fluence2


def load_model_ABmag(filename, dist=100*u.Mpc):
    '''
    Reads in a FITS version of the model template and scales to the given distance.

    Parameters
    ----------

    filename : string
        Path to model FITS.

    Other parameters
    ----------------

    dist : float
        Distance at which to place the source (default is 100*u.Mpc)

    Returns
    -------

    AB magnitude in both DUET bands

    '''

    fitsfile = fits_file(filename)
    model_lc_table = QTable.read(fitsfile)
    model_lc_table['time'] -= model_lc_table['time'][0]
    dist0 = float(model_lc_table.meta['DIST0_PC']) * u.pc

    # Distance modulus
    distmod = (5*np.log10(dist/dist0)).value*u.mag
    ab1 = model_lc_table['mag_D1']*u.ABmag +distmod
    ab2 = model_lc_table['mag_D2']*u.ABmag +distmod

    return model_lc_table['time'], ab1, ab2

def fits_file(file):
    '''
    Helper script to produce the FITS filename
    '''

    sname, ext = os.path.splitext(file)
    if 'fits' in ext:
        return file

    outfile = datadir+'/'+sname+'_lightcurve_DUET.fits'
    if not os.path.exists(outfile):
        log.warn(f"{outfile} does not exist. Creating it now.")
        sims = Simulations()
        sims.parse_emgw()

    return outfile


def load_bai(**kwargs):
    '''Load in the galaxy tables from the Bai catalog

    Returns
    -------
    bai_models : dict
        Contains catalog values.

    '''
    from astroduet.utils import galex_nuv_flux_to_abmag, galex_fuv_flux_to_abmag
    table1 = '../astroduet/data/bai_data/Table1.txt'
    table2 = '../astroduet/data/bai_data/Table2.txt'

    # From http://galex.stsci.edu/gr6/?page=faq
    galex_nuv_bandpass = 732 * u.AA # Effective NUV bandpass
    galex_fuv_bandpass = 268 * u.AA # Effectice FUV bandpass


    f = open(table1, 'r')
    f2 = open(table2, 'r')
    skip = 27
    ctr = 0

    dist = []
    rad= []
    nuv = []
    fuv = []
    pgc = []
    morph = []

    for line in f:
        if ctr < skip:
            ctr += 1
            continue
        else:

            bai_bytes = bytearray(line, 'utf-8')
            pgc = np.append(pgc, int(bai_bytes[0:7]))
            if bai_bytes[59:65] == b'      ':
                thist_dist = -1 * u.Mpc
            else:
                this_dist = float(bai_bytes[59:65])*u.Mpc
            dist = np.append(dist, this_dist)

            # Parse morphology
            if bai_bytes[50:53] == b'   ':
                this_morph = -99
            else:
                this_morph = float(bai_bytes[50:53])
            morph = np.append(morph, this_morph)
    f.close()

    skip = 31
    ctr = 0
    for line in f2:
        if ctr < skip:
            ctr += 1
            continue
        else:

            bai_bytes = bytearray(line, 'utf-8')

            if bai_bytes[52:57] == b'     ':
                this_fuv = -1
            else:
                this_fuv = 10**(float(bai_bytes[52:57]))
            fuv = np.append(fuv, this_fuv)


            if bai_bytes[59:64] == b'     ':
                this_nuv = -1
            else:
                this_nuv = 10**(float(bai_bytes[59:64]))
            nuv = np.append(nuv, this_nuv)



            if bai_bytes[74:80] == b'      ':
                this_rad = -1
            else:
                this_rad = float(bai_bytes[74:80])


            rad = np.append(this_rad, rad)
    #        break
    f.close()

    bai_table = Table(
        [pgc, dist, fuv, nuv, rad, morph],
        names=('PGC', 'DIST', 'LUMFUV', 'LUMNUV', 'RAD', 'MORPH'),
        meta={'name': 'Bai Table 1 and 2'}
        )
    bai_table['RAD'].unit = u.arcsec
    bai_table['DIST'].unit = u.Mpc
    bai_table['LUMNUV'].unit = 'W'
    bai_table['LUMFUV'].unit = 'W'

    good = np.where( (bai_table['LUMNUV'] > 0) & (bai_table['DIST'] > 0) &
                     (bai_table['RAD'] > 0) & (bai_table['MORPH'] > -99) &
                     (bai_table['LUMFUV'] > 0) )
    bai_table = bai_table[good]

    # Surface brightness calculation is here
    bai_table['AREA']= np.pi * (bai_table['RAD']**2)

    # Correct flux estimate?
    flux = (0.5*bai_table['LUMNUV'].to(u.erg / u.s)) / (galex_nuv_bandpass * 4 * np.pi * (bai_table['DIST'].to(u.cm))**2)
    surf_brightness = flux / bai_table['AREA']
    abmag = galex_nuv_flux_to_abmag(surf_brightness) # Now GALEX ABmags per arcsec
    bai_table['SURFNUV'] = abmag

    flux = (0.5*bai_table['LUMFUV'].to(u.erg / u.s)) / (galex_fuv_bandpass * 4 * np.pi * (bai_table['DIST'].to(u.cm))**2)
    surf_brightness = flux / bai_table['AREA']
    abmag = galex_fuv_flux_to_abmag(surf_brightness) # Now GALEX ABmags per arcsec
    bai_table['SURFFUV'] = abmag


    return bai_table