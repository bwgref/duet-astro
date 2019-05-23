from contextlib import contextmanager
import os
import sys

import astropy.units as u
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'data')


@contextmanager
def suppress_stdout():
    """Use context handler to suppress stdout.

    Usage
    -----
    >>> with suppress_stdout():
    ...     print('Bu')
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def duet_abmag_to_fluence_old(ABmag, band, **kwargs):
    """
    Convert AB magnitude for a source into the number of source counts.


    Parameters
    ----------
    ABmag: float
        AB magnitude in the bandpass that you're using

    bandpass: array
        DUET bandpass you're using

    Returns
    -------
    Fluence in the band (ph / cm2 / sec)


    Example
    -------
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> fluence = duet_abmag_to_fluence_old(20*u.ABmag, duet.bandpass1)
    >>> np.isclose(fluence.value, 0.01118276)
    True

    """
    import warnings
    warnings.warn("duet_abmag_to_fluence_old is deprecated; please use"
                  "duet_abmag_to_fluence instead", DeprecationWarning)
    from astropy.modeling.blackbody import FLAM

    import numpy as np

    funit = u.ph / u.cm**2/u.s / u.Angstrom # Spectral radiances per Hz or per angstrom

    bandpass = np.abs( (band[1] - band[0])).to(u.AA)
    midband = np.mean( (band).to(u.AA) )

    fluence = bandpass *  ABmag.to(funit, equivalencies=u.spectral_density(midband))

    return fluence

def duet_fluence_to_abmag_old(fluence, band, **kwargs):
    """
    Convert AB magnitude for a source into the number of source counts.


    Parameters
    ----------
    fluence: float
        fluence in the bandpass that you're using in units (ph / cm2 / sec)

    bandpass: array
        DUET bandpass you're using

    Returns
    -------
    AB magnitude in the band (ABmag)


    Example
    -------
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> funit = u.ph / u.cm**2/u.s
    >>> abmag = duet_fluence_to_abmag_old(0.01*funit, duet.bandpass1)
    >>> np.isclose(abmag.value,  20.12137283)
    True

    """
    import warnings
    warnings.warn("duet_fluence_to_abmag_old is deprecated; please use"
                  "duet_fluence_to_abmag instead", DeprecationWarning)

    bandpass = np.abs( (band[1] - band[0])).to(u.AA)
    midband = np.mean( (band).to(u.AA) )

    ABmag = (fluence / bandpass).to(u.ABmag, equivalencies=u.spectral_density(midband))

    return ABmag


def load_neff():
    """
    Load number of effective background pixels in the PSF from
    file provided by Rick Cook.

    ----
    Returns

    oversample is the ration of the PSF to the pixel size.
    neff is is the resulting value.

    """
    import os
    from numpy import genfromtxt
    from astropy.table import Table
    ref_file = os.path.join(datadir, 'neff_data_full.dat')

    neff_table = Table.read(ref_file, format='ascii')

#    oversig, oversample, neff_center, neff_corner, neff_avg = genfromtxt(ref_file, unpack=True, skip_header=True)
    return neff_table['pix-fwhm'].data, neff_table['avg'].data


def get_neff(psf_size, pixel_size):
    """
    Determine the number of effective background pixels based on the PSF size and the
    pixel size. Assume these are given with astropy units:

    Parameters
    ----------
    psf_size: float
        PSF FWHM size

    pixel-size: float
        Physical size of pixel (in the same units as psf_size)

    Returns
    -------
    The effective number of background pixels that will contribute. Note this is
    fairly idealized, so it's really here as a legacy term.


     Example
    -------
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> neff = get_neff(duet.psf_fwhm, duet.pixel)
    >>> np.isclose(neff, 9.061332801684266)
    True

    """
    from numpy import interp
    over, neff = load_neff()
    data_oversample = (psf_size / pixel_size).value

    neff = interp(data_oversample, over, neff)
    return neff

def galex_to_duet(galmags, duet=None):
    """
    Converts GALEX FUV and NUV ABmags into DUET 1 and DUET 2 ABmags, assuming flat Fnu


    Parameters
    ----------
    galmags: array
        GALEX AB magnitudes, either as [[FUV1, ..., FUVN],[NUV1, ..., NUVN]] or as [[FUV1, NUV1],...,[FUVN, NUVN]]
        Code assumes the first format if len(galmags) = 2

    duet: Telescope instance

    Returns
    -------
    duetmags: Array with same shape as galmags, with DUET 1 and DUET 2 ABmags.

    Example
    -------
    >>> from astroduet.config import Telescope
    >>> duet = Telescope()
    >>> galmags = [20,20]
    >>> duetmags = galex_to_duet(galmags, duet)
    >>> np.allclose(duetmags, [20,20])
    True

    """

    from astropy.modeling.blackbody import FNU

    if duet is None:
        from astroduet.config import Telescope
        duet = Telescope()

    galex_fuv_lef = 151.6 * u.nm
    galex_nuv_lef = 226.7 * u.nm

    duet_1_lef = duet.band1['eff_wave']
    duet_2_lef = duet.band2['eff_wave']

    galex_fuv_nef = galex_fuv_lef.to(u.Hz, u.spectral())
    galex_nuv_nef = galex_nuv_lef.to(u.Hz, u.spectral())

    duet_1_nef = duet_1_lef.to(u.Hz, u.spectral())
    duet_2_nef = duet_2_lef.to(u.Hz, u.spectral())

    # Sort input array into FUV and NUV magnitudes
    if len(galmags) == 2:
        fuv_mag = galmags[0]*u.ABmag
        nuv_mag = galmags[1]*u.ABmag
    else:
        fuv_mag = galmags[:,0]*u.ABmag
        nuv_mag = galmags[:,1]*u.ABmag

    # Convert GALEX magnitudes to flux densities
    fuv_fnu = fuv_mag.to(FNU, u.spectral_density(galex_fuv_nef))
    nuv_fnu = nuv_mag.to(FNU, u.spectral_density(galex_nuv_nef))

    # Extrapolate to DUET bands assuming linear Fnu/nu
    delta_fnu = (nuv_fnu - fuv_fnu)/(galex_nuv_nef - galex_fuv_nef)

    d1_fnu = fuv_fnu + delta_fnu*(duet_1_nef - galex_fuv_nef)
    d2_fnu = fuv_fnu + delta_fnu*(duet_2_nef - galex_fuv_nef)

    # Convert back to magnitudes
    d1_mag = d1_fnu.to(u.ABmag, u.spectral_density(duet_1_nef))
    d2_mag = d2_fnu.to(u.ABmag, u.spectral_density(duet_2_nef))

    # Construct output array
    if len(galmags) == 2:
        duetmags = np.array([d1_mag.value, d2_mag.value])
    else:
        duetmags = np.array([d1_mag.value, d2_mag.value]).transpose()

    return duetmags

def galex_nuv_flux_to_abmag(galflux):
    '''Convert GALEX NUV flux to GALEX NUV ABmag

    Conversion based on this page
    https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html

    Parameters
    ----------
    flux : array
        1D array of GALEX NUV fluxes in units of erg / sec / cm2 / Angstom

    Returns
    -------
    GALEX NUV AB mag:

    '''
    conversion = 2.06e-16


    mAB = (-2.5 * np.log10(galflux.value /conversion) + 20.08)*u.ABmag
    return mAB

def galex_fuv_flux_to_abmag(galflux):
    '''Convert GALEX FUV flux to GALEX FUV ABmag

    Conversion based on this page
    https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html

    Parameters
    ----------
    flux : array
        1D array of GALEX FUV fluxes in units of erg / sec / cm2 / Angstom



    Returns
    -------
    GALEX FUV AB mag:

    '''
    conversion = 1.4e-15
    #    mAB = -2.5 x log10(FluxFUV / 1.40 x 10-15 erg sec-1 cm-2 Å-1) + 18.82

    mAB = (-2.5 * np.log10(galflux.value /conversion) + 18.82)*u.ABmag
    return mAB



def mkdir_p(path):  # pragma: no cover
    """Safe mkdir function.

    Parameters
    ----------
    path : str
        Name of the directory/ies to create

    Notes
    -----
    Found at
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def contiguous_regions(condition):
    """Find contiguous ``True`` regions of the boolean array ``condition``.

    Return a 2D array where the first column is the start index of the region
    and the second column is the end index, found on [so-contiguous]_.

    Parameters
    ----------
    condition : bool array

    Returns
    -------
    idx : ``[[i0_0, i0_1], [i1_0, i1_1], ...]``
        A list of integer couples, with the start and end of each ``True`` blocks
        in the original array

    Notes
    -----
    .. [so-contiguous] http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    """
    # Find the indices of changes in "condition"
    diff = np.logical_xor(condition[1:], condition[:-1])
    idx, = diff.nonzero()
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1
    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]
    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def time_intervals_from_gtis(gtis, chunk_length, fraction_step=1,
                             epsilon=1e-5):
    """Compute start/stop times of equal time intervals, compatible with GTIs.

    Used to start each FFT/PDS/cospectrum from the start of a GTI,
    and stop before the next gap in data (end of GTI).

    Parameters
    ----------
    gtis : 2-d float array
        List of GTIs of the form ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``

    chunk_length : float
        Length of the time segments

    fraction_step : float
        If the step is not a full ``chunk_length`` but less (e.g. a moving window),
        this indicates the ratio between step step and ``chunk_length`` (e.g.
        0.5 means that the window shifts of half ``chunk_length``)

    Returns
    -------
    spectrum_start_times : array-like
        List of starting times to use in the spectral calculations.

    spectrum_stop_times : array-like
        List of end times to use in the spectral calculations.

    """
    spectrum_start_times = np.array([], dtype=np.longdouble)
    for g in gtis:
        if g[1] - g[0] + epsilon < chunk_length:
            continue

        newtimes = np.arange(g[0], g[1] - chunk_length + epsilon,
                             np.longdouble(chunk_length) * fraction_step,
                             dtype=np.longdouble)
        spectrum_start_times = \
            np.append(spectrum_start_times,
                      newtimes)

    assert len(spectrum_start_times) > 0, \
        ("No GTIs are equal to or longer than chunk_length.")
    return spectrum_start_times, spectrum_start_times + chunk_length


def duet_fluence_to_abmag(fluence, duet_no, duet=None, bandpass=None):
    """
    Convert AB magnitude for a source into the number of source counts.


    Parameters
    ----------
    fluence: float
        fluence in the bandpass that you're using in units (ph / cm2 / sec)

    duet_no: int, 1 or 2
        DUET channel

    Other parameters
    ----------------
    duet : `astroduet.config.Telescope` object
        if None, allocate a new Telescope object

    bandpass: array
        DUET bandpass you're using

    Returns
    -------
    AB magnitude in the band (ABmag)


    Example
    -------
    >>> funit = u.ph / u.cm**2/u.s
    >>> abmag = duet_fluence_to_abmag(0.01*funit, 1)
    >>> np.isclose(abmag.value,  18.57586466)
    True

    """
    from astroduet.config import Telescope
    if duet is None:
        duet = Telescope()

    band = getattr(duet, f'band{duet_no}')
    spec = [1] * u.ph / (u.s * u.cm**2 * u.AA)
    wave = [band['eff_wave'].to(u.AA).value] * u.AA
    if bandpass is None:
        bandpass = band['eff_width'].to(u.AA)
    scale = (duet.apply_filters(wave, spec, duet_no)).value[0]

    fluence_corr = fluence / scale

    ABmag = (fluence_corr / bandpass).to(u.ABmag, equivalencies=u.spectral_density(band['eff_wave'].to(u.AA)))

    return ABmag


def duet_abmag_to_fluence(ABmag, duet_no, duet=None, bandpass=None):
    """
    Convert AB magnitude for a source into the number of source counts.


    Parameters
    ----------
    ABmag: float
        AB magnitude in the bandpass that you're using

    duet_no: int, 1 or 2
        DUET channel

    Other parameters
    ----------------
    duet : `astroduet.config.Telescope` object
        if None, allocate a new Telescope object

    bandpass: array
        DUET bandpass you're using

    Returns
    -------
    Fluence in the band (ph / cm2 / sec)


    Example
    -------
    >>> fluence = duet_abmag_to_fluence(20*u.ABmag, 1)
    >>> np.isclose(fluence.value, 0.00269368)
    True

    """
    import numpy as np
    from astroduet.config import Telescope
    if duet is None:
        duet = Telescope()

    band = getattr(duet, f'band{duet_no}')
    spec = [1] * u.ph / (u.s * u.cm**2 * u.AA)
    wave = [band['eff_wave'].to(u.AA).value] * u.AA
    if bandpass is None:
        bandpass = band['eff_width'].to(u.AA)
    scale = (duet.apply_filters(wave, spec, duet_no)).value[0]

    funit = u.ph / u.cm**2/u.s / u.AA # Spectral radiances per Hz or per angstrom

    fluence = bandpass * ABmag.to(funit,
                                  equivalencies=u.spectral_density(band['eff_wave'].to(u.AA)))

    return fluence * scale

