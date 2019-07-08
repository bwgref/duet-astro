import os
import copy
import pickle

import numpy as np

from scipy.interpolate import interp1d

from astropy.table import Table, QTable
from astropy import log
import astropy.units as u
from .duet_sensitivity import calc_snr
from .utils import get_neff, suppress_stdout, mkdir_p
from .utils import tqdm as imported_tqdm
from .utils import contiguous_regions, duet_fluence_to_abmag
from .bbmag import sigerr
from .config import Telescope
from .background import background_pixel_rate
from .image_utils import construct_image, run_daophot, find, estimate_background
from .models import load_model_fluence, load_model_ABmag
from .diff_image import calculate_diff_image


def join_equal_gti_boundaries(gti):
    """If the start of a GTI is right at the end of another, join them.

    """
    new_gtis = gti
    touching = gti[:-1, 1] == gti[1:, 0]
    if np.any(touching):
        ng = []
        count = 0
        while count < len(gti) - 1:
            if new_gtis[count, 1] == gti[count + 1, 0]:
                ng.append([gti[count, 0], gti[count + 1, 1]])
            else:
                ng.append(gti[count])
            count += 1
        new_gtis = np.asarray(ng)
    return new_gtis


def cross_two_gtis(gti0, gti1):
    """Extract the common intervals from two GTI lists *EXACTLY*.

    From Stingray

    Parameters
    ----------
    gti0 : iterable of the form ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
    gti1 : iterable of the form ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        The two lists of GTIs to be crossed.

    Returns
    -------
    gtis : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        The newly created GTIs

    See Also
    --------
    cross_gtis : From multiple GTI lists, extract common intervals *EXACTLY*

    Examples
    --------
    >>> gti1 = np.array([[1, 2]])
    >>> gti2 = np.array([[1, 2]])
    >>> newgti = cross_two_gtis(gti1, gti2)
    >>> np.all(newgti == [[1, 2]])
    True
    >>> gti1 = np.array([[1, 4]])
    >>> gti2 = np.array([[1, 2], [2, 4]])
    >>> newgti = cross_two_gtis(gti1, gti2)
    >>> np.all(newgti == [[1, 4]])
    True
    """
    gti0 = join_equal_gti_boundaries(np.asarray(gti0))
    gti1 = join_equal_gti_boundaries(np.asarray(gti1))
#     # Check GTIs
#     check_gtis(gti0)
#     check_gtis(gti1)

    gti0_start = gti0[:, 0]
    gti0_end = gti0[:, 1]
    gti1_start = gti1[:, 0]
    gti1_end = gti1[:, 1]

    # Create a list that references to the two start and end series
    gti_start = [gti0_start, gti1_start]
    gti_end = [gti0_end, gti1_end]

    # Concatenate the series, while keeping track of the correct origin of
    # each start and end time
    gti0_tag = np.array([0 for g in gti0_start], dtype=bool)
    gti1_tag = np.array([1 for g in gti1_start], dtype=bool)
    conc_start = np.concatenate((gti0_start, gti1_start))
    conc_end = np.concatenate((gti0_end, gti1_end))
    conc_tag = np.concatenate((gti0_tag, gti1_tag))

    # Put in time order
    order = np.argsort(conc_end)
    conc_start = conc_start[order]
    conc_end = conc_end[order]
    conc_tag = conc_tag[order]

    last_end = conc_start[0] - 1
    final_gti = []
    for ie, e in enumerate(conc_end):
        # Is this ending in series 0 or 1?
        this_series = int(conc_tag[ie])
        other_series = int(this_series == 0)

        # Check that this closes intervals in both series.
        # 1. Check that there is an opening in both series 0 and 1 lower than e
        try:
            st_pos = \
                np.argmax(gti_start[this_series][gti_start[this_series] < e])
            so_pos = \
                np.argmax(gti_start[other_series][gti_start[other_series] < e])
            st = gti_start[this_series][st_pos]
            so = gti_start[other_series][so_pos]

            s = np.max([st, so])
        except:  # pragma: no cover
            continue

        # If this start is inside the last interval (It can happen for equal
        # GTI start times between the two series), then skip!
        if s <= last_end:
            continue
        # 2. Check that there is no closing before e in the "other series",
        # from intervals starting either after s, or starting and ending
        # between the last closed interval and this one
        cond1 = (gti_end[other_series] > s) * (gti_end[other_series] < e)
        cond2 = gti_end[other_series][so_pos] < s
        condition = np.any(np.logical_or(cond1, cond2))
        # Well, if none of the conditions at point 2 apply, then you can
        # create the new gti!
        if not condition:
            final_gti.append([s, e])
            last_end = e

    return np.array(final_gti)


def get_visibility_windows(observation_start : float, observation_end : float,
                           orbital_period : float = 96. * 60,
                           exposure_per_orbit : float = 35. * 60,
                           phase_start : float = 0.):
    """Observing windows of a given target over multiple orbits.

    All quantities are defined in seconds.
    This function is unit-agnostic.

    Parameters
    ----------
    observation_start : float
        Observation start in seconds
    observation_end : float
        Observation end in seconds

    Other parameters
    ----------------
    orbital_period : float, default 96 mins
        Orbital period of the satellite, in seconds
    exposure_per_orbit : float, default 35 mins
        Time spent on a given target during an orbit, in seconds
    phase_start : float, default 0
        Orbital phase at which a target is observed


    Examples
    --------
    >>> ow = get_visibility_windows(0, 5760 * 2)
    >>> np.allclose(ow, [[0, 2100], [5760, 7860]])
    True
    """
    tstart = observation_start + orbital_period * phase_start
    start_times = np.arange(tstart, observation_end + orbital_period, orbital_period)
    end_times = start_times + exposure_per_orbit
    obs_windows = np.array(list(zip(start_times, end_times)))
    good = (start_times < observation_end)
    obs_windows[end_times > observation_end] = observation_end

    return obs_windows[good]


def calculate_flux(time : float, flux : float):
    """This is unit-agnostic, by choice.

    Examples
    --------
    >>> time = np.arange(10)
    >>> flux = np.ones(10) * 30
    >>> np.isclose(calculate_flux(time, flux), 30)
    True
    """
    dt = np.mean(np.diff(time))
    return np.sum(flux * dt) / (time[-1] - time[0] + dt)


def calculate_lightcurve_from_model(model_time, model_lc,
                                    observing_windows=None,
                                    visibility_windows=None,
                                    exposure_length=300 * u.s, **kwargs):
    """Calculate a light curve from the model lightcurve in ct/s.

    Parameters
    ----------
    model_time : ``astropy.units.s``
        Times at which the model is calculated
    model_lc : any ``astropy.units`` object expressing flux
        Values of the model

    Other parameters
    ----------------
    visibility_windows : [[start0, end0], [start1, end1], ...], ``astropy.units.s``, default None
        Visibility windows in seconds
    observing_windows : [[start0, end0], [start1, end1], ...], ``astropy.units.s``, default None
        Observing times in seconds
    exposure_length : ``astropy.units.s``, default 300 s
        Time for each exposure
    **kwargs :
        Additional keyword arguments to be passed to
        ``get_observing_windows``
    """
    observing_windows = np.array([[model_time[0].to(u.s).value,
                                   model_time[-1].to(u.s).value]]) * u.s \
        if observing_windows is None else observing_windows

    if visibility_windows is None:
        visibility_windows = \
            get_visibility_windows(observing_windows.min().to(u.s).value,
                                   observing_windows.max().to(u.s).value,
                                   **kwargs) * u.s

    observing_windows = cross_two_gtis(observing_windows,
                                       visibility_windows) * u.s

    interpolated_lc = interp1d(model_time.to(u.s).value,
                               model_lc.value, fill_value=0,
                               bounds_error=False)
    times = []
    lc = []
    for ow in observing_windows:
        expo_times = np.arange(ow[0].value, ow[1].value,
                               exposure_length.value) * u.s
        for t in expo_times:
            times.append((t + exposure_length / 2).to(u.s).value)
            fine_times = np.linspace(t.value, (t + exposure_length).value, 10)
            fine_model = interpolated_lc(fine_times)
            flux = calculate_flux(fine_times * u.s, fine_model)
            lc.append(flux)

    result_table = QTable()
    result_table['time'] = np.array(times) * u.s
    result_table['Light curve'] = np.array(lc) * model_lc.unit

    return result_table


def calculate_snr(duet, band_fluence,
                  texp=300*u.s, read_noise=3. * (2 ** 0.5),
                  background=0):
    """Calculate the signal to noise ratio of from photon fluxes.

    Parameters
    ----------
    duet : a ``astroduet.config.Telescope`` instance
    band_fluence : array of floats
        Series of photon flux measurements, rescaled at 10 pc

    Other parameters
    ----------------
    background : float
        Background level
    texp : float
        Exposure time
    read_noise : float, default 3\\sqrt(2)
        Readout noise
    """
    neff = get_neff(duet.psf_size, duet.pixel)
    efficiency = (duet.eff_epd / duet.EPD)**2
    area = np.pi * duet.EPD**2

    band_rate = duet.trans_eff * efficiency * area * band_fluence
    lc_snr = calc_snr(texp, band_rate, background,
                      read_noise, neff)
    return lc_snr


def get_lightcurve(input_lc_file, distance=10*u.pc, observing_windows=None,
                   duet=None, low_zodi=True, exposure=300 * u.s,
                   **kwargs):
    """Get a realistic light curve from a given theoretical light curve.

    Parameters
    ----------
    input_lc_file : str
        Light curve file containing photon fluxes

    Other parameters
    ----------------
    distance : ``astropy.units.pc``
        Distance of the SN event
    observing_windows : [[start0, end0], [start1, end1], ...], ``astropy.units.s``, default None
        Observing times in seconds
    duet : ``astroduet.config.Telescope`` object
        The telescope config to be used. Default config if None.
    low_zodi : bool, default True
        Low zodiacal light?
    exposure : ``astropy.units.s``, default 300 s
        Exposure time
    """
    if duet is None:
        duet = Telescope()

    model_lc_table_fl = QTable(load_model_fluence(input_lc_file,
                                                  dist=distance))
    abtime, ab1, ab2 = load_model_ABmag(input_lc_file,
                                        dist=distance)
    model_lc_table_ab = QTable({'time': abtime, 'mag_D1': ab1, 'mag_D2':ab2})

    assert np.allclose(model_lc_table_fl['time'].value,
                       model_lc_table_ab['time'].value)
    model_lc_table = model_lc_table_fl
    for col in ['mag_D1', 'mag_D2']:
        model_lc_table[col] = model_lc_table_ab[col]

    result_table = QTable()

    background = background_pixel_rate(duet, low_zodi=low_zodi)
    background = {'bkg_D1': background[0], 'bkg_D2': background[1]}

    for duet_no in range(1, 3):
        duet_label = f'D{duet_no}'

        table_photflux = \
            calculate_lightcurve_from_model(
                model_lc_table['time'],
                model_lc_table[f'fluence_{duet_label}'],
                exposure_length=exposure,
                observing_windows=observing_windows,
                **kwargs)

        table_ab = \
            calculate_lightcurve_from_model(
                model_lc_table['time'],
                model_lc_table[f'mag_{duet_label}'],
                exposure_length=exposure,
                observing_windows=observing_windows,
                **kwargs)

        if duet_no == 1:
            result_table['time'] = table_photflux['time']

        result_table[f'fluence_{duet_label}'] = \
            table_photflux['Light curve'].to(u.ph / u.cm**2 / u.s)

        rate = duet.fluence_to_rate(result_table[f'fluence_{duet_label}'])

        result_table[f'snr_{duet_label}'] = \
            duet.calc_snr(exposure, rate, background[f'bkg_{duet_label}']) \
                * u.dimensionless_unscaled

        abmag = table_ab['Light curve']
        abmag_err = sigerr(abmag)
        result_table[f'mag_{duet_label}'] = abmag
        result_table[f'mag_{duet_label}_err'] =  abmag_err

    return result_table


def continuous_time_intervals(lightcurve, dt=None, tolerance=None):
    """Find points in the lightcurve with two or more contiguous measurements.

    Examples
    --------
    >>> times = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
    >>> lightcurve = QTable({'time': times * u.s})
    >>> intvs = continuous_time_intervals(lightcurve)
    >>> np.allclose(intvs.value, np.array([[-0.5, 2.5], [3.5, 10.5]]))
    True
    >>> times = np.array([0, 2, 4, 5, 6, 7, 8, 9, 10])
    >>> lightcurve = QTable({'time': times * u.s})
    >>> intvs = continuous_time_intervals(lightcurve)
    >>> np.allclose(intvs.value,
    ...             np.array([[3.5, 10.5]]))
    True
    """
    times = lightcurve['time']
    diff = np.diff(times)
    if dt is None:
        dt = np.median(diff)
    if tolerance is None:
        tolerance = dt

    good = np.abs(diff - dt < tolerance)
    good_bins = contiguous_regions(good)
    good_time_bins = times[good_bins]
    good_time_bins[:, 0] -= dt
    good_time_bins += dt / 2
    return good_time_bins


def _decide_bins_to_group(lightcurve, exposure, final_resolution,
                          tolerance=None):
    """
    Examples
    --------
    >>> times = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
    >>> lightcurve = QTable({'time': times * u.s})
    >>> newlc = _decide_bins_to_group(lightcurve, 1 * u.s, 4 * u.s)
    >>> time_bins = newlc['time_bin']
    >>> np.allclose(time_bins, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    True
    """
    gti = continuous_time_intervals(lightcurve, exposure, tolerance=tolerance)

    start = gti[:, 0]
    end = gti[:, 1]
    current_start_bin = 0
    current_time_bin = 0
    plain_lc = copy.deepcopy(lightcurve)
    plain_lc['time_bin'] = -10

    last_interval_bin = 0
    for i, t in enumerate(plain_lc['time']):
        if t < np.min(start):
            continue
        if t > np.max(end):
            continue
        start_bin = np.searchsorted(start, t) - 1
        if start_bin > current_start_bin:
            current_time_bin += 1
            current_start_bin = start_bin
            last_interval_bin = current_time_bin
        current_time_bin = last_interval_bin + ((t - start[start_bin]) // final_resolution).to("").value
        plain_lc['time_bin'][i] = current_time_bin

    return plain_lc


def rebin_lightcurve(lightcurve, exposure, final_resolution, debug=False, debugfilename='lightcurve_rebin.hdf5'):
    """
    Examples
    --------
    >>> times = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
    >>> flu_units = u.ph / u.s / u.cm**2
    >>> lightcurve = QTable({'time': times * u.s,
    ...                      'fluence_D1': np.ones(len(times)) * flu_units,
    ...                      'fluence_D2': np.ones(len(times)) * flu_units})
    >>> frame = [30, 30]
    >>> lightcurve['imgs_D1'] = np.ones(
    ...     (len(lightcurve), frame[0], frame[1])) * u.ph / u.s
    >>> lightcurve['imgs_D2'] = np.zeros(
    ...     (len(lightcurve), frame[0], frame[1])) * u.ph / u.s
    >>> lightcurve['imgs_D1_bkgsub'] = np.zeros(
    ...     (len(lightcurve), frame[0], frame[1])) * u.ph / u.s
    >>> lightcurve['imgs_D2_bkgsub'] = np.zeros(
    ...     (len(lightcurve), frame[0], frame[1])) * u.ph / u.s
    >>> newlc = rebin_lightcurve(lightcurve, 1 * u.s, 4 * u.s)
    >>> np.allclose(newlc['time'].value, np.array([1, 5.5, 9]))
    True
    >>> np.allclose(newlc['fluence_D1'].value, np.array([1, 1, 1]))
    True
    >>> np.allclose(newlc['imgs_D1'].value.sum(axis=(1,2)) / 900,
    ...             np.array([1, 1, 1]))
    True
    >>> np.allclose(newlc['imgs_D2'].value.sum(axis=(1,2)),
    ...             np.array([0, 0, 0]))
    True
    """
    new_lightcurve = QTable()
    lightcurve = \
        _decide_bins_to_group(lightcurve, exposure,
                              final_resolution,
                              tolerance=final_resolution)
    time_bin = lightcurve['time_bin']
    good = time_bin >= 0
    plain_lc = Table(copy.deepcopy(lightcurve[good]))
    plain_lc['nbin'] = 1
    # time_bin = (plain_lc['time'] // final_resolution).to("").value
    lc_group = plain_lc.group_by(time_bin)
    plain_lc = lc_group.groups.aggregate(np.sum)
    cols = 'time,fluence_D1,fluence_D2'.split(',')
    if 'imgs_D1' in plain_lc.colnames:
        cols += 'imgs_D1,imgs_D2,imgs_D1_bkgsub,imgs_D2_bkgsub'.split(',')
    for col in cols:
        new_lightcurve[col] = [value / nb
                               for value, nb in zip(plain_lc[col],
                                                    plain_lc['nbin'])] * lightcurve[col].unit
    new_lightcurve['nbin'] = plain_lc['nbin']
    if debug:
        new_lightcurve.write(debugfilename, path='default',
                             overwrite=True)
    return new_lightcurve


def construct_images_from_lightcurve(lightcurve, exposure, duet=None,
                                     gal_type=None,
                                     gal_params=None,
                                     frame=np.array([30, 30]),
                                     debug=False,
                                     debugfilename='lightcurve.hdf5',
                                     zodi='low',
                                     silent=False):
    """
    Examples
    --------
    >>> import doctest
    >>> doctest.ELLIPSIS_MARKER = '-etc-'
    >>> times = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10])
    >>> flu_units = u.ph / u.s / u.cm**2
    >>> lightcurve = QTable({'time': times * u.s,
    ...                      'fluence_D1': np.ones(len(times)) * flu_units,
    ...                      'fluence_D2': np.ones(len(times)) * flu_units})
    >>> lc_new = construct_images_from_lightcurve(lightcurve, 1 * u.s) # doctest:+ELLIPSIS
    -etc-
    >>> np.allclose(lc_new['time'].value, lightcurve['time'].value)
    True
    >>> 'imgs_D1' in lightcurve.colnames
    True
    """
    if duet is None:
        with suppress_stdout():
            duet = Telescope()
    if silent:
        tqdm = lambda x: x
    else:
        tqdm = imported_tqdm

    with suppress_stdout():
        if zodi == 'low':
            [bgd_band1, bgd_band2] = background_pixel_rate(duet,
                                                           low_zodi=True,
                                                           diag=True)
        elif zodi == 'med':
            [bgd_band1, bgd_band2] = background_pixel_rate(duet,
                                                           med_zodi=True,
                                                           diag=True)
        elif zodi == 'high':
            [bgd_band1, bgd_band2] = background_pixel_rate(duet,
                                                           high_zodi=True,
                                                           diag=True)

    lightcurve['imgs_D1'] = \
        np.zeros((len(lightcurve), frame[0], frame[1])) * u.ph / u.s
    lightcurve['imgs_D2'] = \
        np.zeros((len(lightcurve), frame[0], frame[1])) * u.ph / u.s

    lightcurve['imgs_D1_bkgsub'] = \
        np.zeros((len(lightcurve), frame[0], frame[1])) * u.ph / u.s
    lightcurve['imgs_D2_bkgsub'] = \
        np.zeros((len(lightcurve), frame[0], frame[1])) * u.ph / u.s

    if not silent:
        log.info('Creating images')
    for i, row in enumerate(tqdm(lightcurve)):
        fl1 = duet.fluence_to_rate(row['fluence_D1'])
        fl2 = duet.fluence_to_rate(row['fluence_D2'])

        with suppress_stdout():
            image1 = construct_image(frame, exposure, duet=duet,
                                     band=duet.bandpass1,
                                     source=fl1, gal_type=gal_type,
                                     gal_params=gal_params,
                                     sky_rate=bgd_band1)
        image_rate1 = image1 / exposure
        image_bkg, image_bkg_rms_median = \
            estimate_background(image_rate1, method='1D', sigma=2)
        image_rate_bkgsub = image_rate1 - image_bkg

        lightcurve['imgs_D1'][i] = image_rate1
        lightcurve['imgs_D1_bkgsub'][i] = image_rate_bkgsub

        with suppress_stdout():
            image2 = construct_image(frame, exposure, duet=duet, band=duet.bandpass2,
                                     source=fl2, gal_type=gal_type, gal_params=gal_params,
                                     sky_rate=bgd_band2)
        image_rate2 = image2 / exposure
        image_bkg, image_bkg_rms_median = \
            estimate_background(image_rate2, method='1D', sigma=2)
        image_rate_bkgsub = image_rate2 - image_bkg

        lightcurve['imgs_D2'][i] = image_rate2
        lightcurve['imgs_D2_bkgsub'][i] = image_rate_bkgsub

    if debug:
        outfile = debugfilename
        if outfile.endswith('.hdf5'):
            lightcurve.write(outfile, overwrite=True, path='default')
        else:
            lightcurve.write(outfile, overwrite=True)

    return lightcurve


def lightcurve_through_image(lightcurve, exposure,
                             frame=np.array([30, 30]),
                             final_resolution=None,
                             duet=None,
                             gal_type=None, gal_params=None,
                             debug=False, debugfilename='lightcurve',
                             silent=False,zodi='low',
                             ignore_low_check=False):
    """Transform a theoretical light curve into a flux measurement.

    1. Take the values of a light curve, optionally rebin it to a new time
    resolution.
    2. Then, create an image with a point source corresponding to each flux
    measurement, and calculate the flux from the image with ``daophot``.
    3. Return the ''realistic'' light curve

    Parameters
    ----------
    lightcurve : ``astropy.table.Table``
        The lightcurve has to contain the columns 'time', 'fluence_D1', and
        'fluence_D2'. Photon fluxes are in counts/s.
    exposure : ``astropy.units.Quantity``
        Exposure time used for the light curve

    Other parameters
    ----------------
    frame : [N, M]
        Number of pixel along x and y axis
    final_resolution : ``astropy.units.Quantity``, default None
        Rebin the light curve to this time resolution before creating the light
         curve. Must be > exposure
    duet : ``astroduet.config.Telescope``
        If None, a default one is created
    gal_type : string
        Default galaxy string ("spiral"/"elliptical") or "custom" w/ Sersic parameters
        in gal_params
    gal_params : dict
        Dictionary of parameters for Sersic model (see sim_galaxy)
    debug : bool
        If True, save the light curves to file
    debugfilename : str
        File to save the light curves to
    silent : bool
        Suppress progress bars
    zodi : string
        Either 'low', 'med', or 'high'

    Returns
    -------
    lightcurve : ``astropy.table.Table``
        A light curve, rebinned to ``final_resolution``, and with four new
        columns: 'fluence_D1_fit', 'fluence_D1_fiterr', 'fluence_D2_fit',
        and 'fluence_D2_fiterr', containing the flux measurements from the
        intermediate images and their errorbars.
    """
    from astropy.table import Table
    lightcurve = copy.deepcopy(lightcurve)
    if silent:
        tqdm = lambda x: x
    else:
        tqdm = imported_tqdm

    with suppress_stdout():
        if duet is None:
            duet = Telescope()

    with suppress_stdout():
        if zodi == 'low':
            [bgd_band1, bgd_band2] = background_pixel_rate(duet,
                                                           low_zodi=True,
                                                           diag=True)
        elif zodi == 'med':
            [bgd_band1, bgd_band2] = background_pixel_rate(duet,
                                                           med_zodi=True,
                                                           diag=True)
        elif zodi == 'high':
            [bgd_band1, bgd_band2] = background_pixel_rate(duet,
                                                           high_zodi=True,
                                                           diag=True)

    # Directory for debugging purposes
    if debugfilename != 'lightcurve':
        debugdir = os.path.join('debug_imgs',debugfilename)
    else:
        rand = np.random.randint(0, 99999999)
        debugdir = f'debug_imgs_{rand}'

    if debug:
        mkdir_p(debugdir)

    good = (lightcurve['fluence_D1'] > 0) & (lightcurve['fluence_D2'] > 0)
    if not ignore_low_check:
        if not np.any(good):
            log.warning("Light curve has no points with fluence > 0")
            return
        lightcurve = lightcurve[good]

    lightcurve = \
        construct_images_from_lightcurve(
            lightcurve, exposure, duet=duet, gal_type=gal_type,
            gal_params=gal_params, frame=frame, debug=debug,
            debugfilename=os.path.join(debugdir, debugfilename+'.hdf5'),
            zodi=zodi, silent=silent)

    total_image_rate1 = np.sum(lightcurve['imgs_D1'], axis=0)
    total_image_rate2 = np.sum(lightcurve['imgs_D2'], axis=0)

    total_image_rate = total_image_rate1 + total_image_rate2

    total_image_rate_bkgsub1 = np.sum(lightcurve['imgs_D1_bkgsub'], axis=0)
    total_image_rate_bkgsub2 = np.sum(lightcurve['imgs_D2_bkgsub'], axis=0)

    total_image_rate_bkgsub = \
        total_image_rate_bkgsub1 + total_image_rate_bkgsub2

    total_images_rate_list = [total_image_rate1, total_image_rate2]

    psf_fwhm_pix = duet.psf_fwhm / duet.pixel

    log.info('Constructing reference images')
    # Make reference images (5 exposures)
    if final_resolution is not None:
        nexp = int(np.rint(5 * final_resolution / exposure))
    else:
        nexp = 5

    ref_image1 = construct_image(frame, exposure, duet=duet,
                                 band=duet.bandpass1, gal_type=gal_type,
                                 gal_params=gal_params, sky_rate=bgd_band1,
                                 n_exp=nexp)
    ref_image_rate1 = ref_image1 / (exposure * nexp)
    ref_bkg1, ref_bkg_rms_median1 = estimate_background(ref_image_rate1, method='1D', sigma=2)
    ref_rate_bkgsub1 = ref_image_rate1 - ref_bkg1

    ref_image2 = construct_image(frame, exposure, duet=duet,
                                 band=duet.bandpass2, gal_type=gal_type,
                                 gal_params=gal_params, sky_rate=bgd_band2,
                                 n_exp=nexp)
    ref_image_rate2 = ref_image2 / (exposure * nexp)
    ref_bkg2, ref_bkg_rms_median2 = estimate_background(ref_image_rate2, method='1D', sigma=2)
    ref_rate_bkgsub2 = ref_image_rate2 - ref_bkg2

    # psf_array = duet.psf_model(x_size=5,y_size=5).array

    total_ref_img_rate = ref_image_rate1 + ref_image_rate2
    total_ref_img_rate_bkgsub = \
        ref_rate_bkgsub1 + ref_rate_bkgsub2

    log.info('Finding source in integrated diff image')
    diff_total_image = \
        calculate_diff_image(total_image_rate, total_image_rate_bkgsub,
                             total_ref_img_rate,
                             total_ref_img_rate_bkgsub,
                             duet)

    with suppress_stdout():
        star_tbl, bkg_image, threshold = find(diff_total_image,
                                              psf_fwhm_pix.value,
                                              method='daophot',
                                              background='1D', frame='diff')
    if len(star_tbl) < 1:
        log.warning("No good detections in this field")
        return

    if debug:
        outfile = os.path.join(debugdir, f'images_total.p')
        with open(outfile, 'wb') as fobj:
            pickle.dump({'imgD1': total_images_rate_list[0],
                         'imgD2': total_images_rate_list[1]},
                        fobj)
        outfile = os.path.join(debugdir, f'images_ref.p')
        with open(outfile, 'wb') as fobj:
            pickle.dump({'imgD1': ref_image_rate1, 'imgD2': ref_image_rate2},
                        fobj)

        outfile = os.path.join(debugdir, f'images_diff.p')
        with open(outfile, 'wb') as fobj:
            pickle.dump({'imgD1': diff_total_image},
                        fobj)

    star_tbl.sort('flux')
    star_tbl = star_tbl[-1:]['x', 'y']

    # decide light curve bins before image generation, for speed.
    lightcurve['nbin'] = 1
    if final_resolution is not None:
        lightcurve = rebin_lightcurve(lightcurve, exposure, final_resolution,
                                      debug=debug)

    for duet_no in [1, 2]:
        for suffix in ['', 'err']:
            colname = f'fluence_D{duet_no}_fit{suffix}'
            lightcurve[colname] = 0.
            lightcurve[colname].unit = u.ph / (u.cm**2 * u.s)
            colname = f'ABmag_D{duet_no}_fit{suffix}'
            lightcurve[colname] = 0.

    lightcurve['snr_D1'] = 0.
    lightcurve['snr_D2'] = 0.
    # Generate light curve
    log.info('Measuring fluxes and creating light curve')
    for i, row in enumerate(tqdm(lightcurve)):
        time = row['time']

        image_rate1 = lightcurve['imgs_D1'][i]
        image_rate_bkgsub1 = lightcurve['imgs_D1_bkgsub'][i]

        diff_image1 = calculate_diff_image(image_rate1, image_rate_bkgsub1,
                                           ref_image_rate1, ref_rate_bkgsub1,
                                           duet=duet)

        with suppress_stdout():
            result1, _ = run_daophot(diff_image1, threshold,
                                     star_tbl, niters=1, snr_lim=0., duet=duet)

        fl1_fit = result1['flux_fit'][0] * image_rate1.unit
        fl1_fite = result1['flux_unc'][0] * image_rate1.unit
        lightcurve['fluence_D1_fit'][i] = duet.rate_to_fluence(fl1_fit)
        lightcurve['fluence_D1_fiterr'][i] = duet.rate_to_fluence(fl1_fite)
        if (fl1_fit > 0) & (fl1_fite > 0):
            lightcurve['snr_D1'][i] = fl1_fit / fl1_fite
            lightcurve['ABmag_D1_fit'][i] = \
                duet_fluence_to_abmag(lightcurve['fluence_D1_fit'][i], 1,
                                      duet=duet).value
            ABerr = 2.5 * np.log(1 + 1 / lightcurve['snr_D1'][i])
            lightcurve['ABmag_D1_fiterr'][i] = ABerr

        image_rate2 = lightcurve['imgs_D2'][i]
        image_rate_bkgsub2 = lightcurve['imgs_D2_bkgsub'][i]

        diff_image2 = calculate_diff_image(image_rate2, image_rate_bkgsub2,
                                           ref_image_rate2, ref_rate_bkgsub2,
                                           duet=duet)

        with suppress_stdout():
            result2, _ = run_daophot(diff_image2, threshold,
                                     star_tbl, niters=1, snr_lim=0., duet=duet)
        fl2_fit = result2['flux_fit'][0] * image_rate2.unit
        fl2_fite = result2['flux_unc'][0] * image_rate2.unit

        lightcurve['fluence_D2_fit'][i] = duet.rate_to_fluence(fl2_fit)
        lightcurve['fluence_D2_fiterr'][i] = duet.rate_to_fluence(fl2_fite)
        if (fl2_fit > 0) & (fl2_fite > 0):
            lightcurve['snr_D2'][i] = fl2_fit / fl2_fite
            lightcurve['ABmag_D2_fit'][i] = \
                duet_fluence_to_abmag(lightcurve['fluence_D2_fit'][i], 2,
                                      duet=duet).value
            ABerr = 2.5 * np.log(1 + 1 / lightcurve['snr_D2'][i])
            lightcurve['ABmag_D2_fiterr'][i] = ABerr

    return lightcurve
