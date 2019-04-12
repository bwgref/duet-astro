import os
from astroduet.lightcurve import get_lightcurve

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, '..', 'data')


def test_load_lightcurve():
    # Dummy test
    lc = get_lightcurve(os.path.join(datadir, "SNIIb_lightcurve_DUET.fits"))
    assert 'time' in lc.colnames
