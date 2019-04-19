import os
from astroduet.lightcurve import get_lightcurve, lightcurve_through_image
from astropy.table import Table
import astropy.units as u

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, '..', 'data')


def test_load_lightcurve():
    # Dummy test
    lc = get_lightcurve(os.path.join(datadir, "SNIIb_lightcurve_DUET.fits"))
    assert 'time' in lc.colnames


def test_realistic_lightcurve():
    # Dummy test
    lc = Table({'time': [150, 450] * u.s,
                'photflux_D1':  [4, 6] * (1 / u.cm**2 / u.s),
                'photflux_D2': [3, 1] * (1 / u.cm**2 / u.s)})
    lc_out = lightcurve_through_image(lc, 300 * u.s,
                                      final_resolution=600 * u.s)
    assert 'photflux_D1_fiterr' in lc_out.colnames
    assert len(lc_out['time']) == 1
