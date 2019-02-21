import glob
import os
import numpy as np


def wavelength_to_energy(wave):
    """

    """
    from astropy import units as u
    from astropy import constants as c
    return (c.h * c.c / wave).to(u.Joule)


def rebin_spectrum(wave, flux, new_wave_grid):
    from scipy.stats import binned_statistic
    stat, _, _ = binned_statistic(wave, flux, statistic='mean',
                            bins=new_wave_grid)
    return stat


def load_zodi(**kwargs):
    """
    From here
    https://cads.iiap.res.in/tools/zodiacalCalc/Documentation

    They provied the tabulated Zodi spectrum:
    https://cads.iiap.res.in/download/simulations/scaled_zodiacal_spec.txt

    From here
    Colina et al
    http://adsabs.harvard.edu/abs/1996AJ....112..307C

    According to this, the flux has been scaled so that at 5000 Ang
    the flux has units of 252 W / m2 / sr / micon

    This takes as input the flux density as read off from Table 17 of
    https://aas.aanda.org/articles/aas/pdf/1998/01/ds1449.pdf

    scale = scale in units of [1e-8 W / m2 / sr / micron at 500 nm]
    Default is for polar zodiacal emission, which is 77 in the above units.

    Toward the ecliptic plane this number can grow to be >1000

    For a Sun avoidance of 45 degrees this looks like a value of 200 - 900
    based strongly on the heliocentric longitdue. However, if you try
    72, 300, and 1000 it looks like you'll probably span this space.

    Examples
    --------
    >>> a = load_zodi()

    """
    from astropy import units as ur

    scale = kwargs.pop('scale', 77)

    ftab_unit = ur.W/ur.m**2 / ur.micron / ur.sr

    scale_norm = False
    wave, flux = np.genfromtxt(os.path.join('input_data',
                                            'scaled_zodiacal_spec.txt'),
                               unpack=True)

    spec = {'wavelength': wave * ur.Angstrom, 'flux': flux}
    if not scale_norm:
        wave_at_5e3 = np.argmin(np.abs(wave - 5e3))
        scale_norm = float(flux[wave_at_5e3])

    # Normalize to flux density at 500 nm:
    spec['flux'] = spec["flux"] * (scale*1e-8 / scale_norm) * ftab_unit

    for airglow_spec in glob.glob(os.path.join("input_data", "airglow*.dat")):
        # read spectrum
        wave, flux = np.genfromtxt(airglow_spec,
                                   unpack=True, comments=";")
        
        # rebin it to 1 Angstrom. The choice of +- 0.5 is because of how
        # binned_statistics works
        w0 = np.floor(wave.min())
        w1 = np.ceil(wave.max()) + 1
        new_wave_grid = np.arange(w0 - 0.5, w1 + 0.5, 1)
        new_flux = rebin_spectrum(wave, flux, new_wave_grid)
        # back to wanted grid (half bins of grid used in rebin_spectrum)
        new_wave_grid = np.arange(w0, w1, 1)

        # eliminate NaNs
        good = (new_flux == new_flux)

        # give units!
        new_wave_grid = new_wave_grid[good] * ur.Angstrom
        photon_to_power = wavelength_to_energy(new_wave_grid) / ur.s

        # Need to work around Astropy bug with units, guiding the conversion
        new_flux = new_flux[good] * photon_to_power / ur.A / ur.cm**2 / ur.sr
        new_flux *= ur.W / (ur.J / ur.s)
        new_flux *= ur.A / (ur.um)
        new_flux *= ur.cm**2 / (ur.m**2)

        # look for the wanted spectral bin and sum the new flux
        for w, f in zip(new_wave_grid, new_flux):
            idx = np.argmin(np.abs(spec['wavelength'] - w))
            spec["flux"][idx] += f

    return spec
