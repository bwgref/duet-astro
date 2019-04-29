import os
import glob
import shutil
from astroduet.lightcurve import get_lightcurve, lightcurve_through_image
from astropy.table import Table, QTable
import astropy.units as u
import pytest

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, '..', 'data')

try:
    import h5py
    HAS_H5PY = True
except:
    HAS_H5PY = False


def test_load_lightcurve():
    # Dummy test
    lc = get_lightcurve("shock_2.5e10.dat")
    assert 'time' in lc.colnames


pytest.mark.skipif('not HAS_H5PY')
def test_realistic_lightcurve():
    # Dummy test
    debug_img_dir = glob.glob('debug_imgs_*')
    if len(debug_img_dir) > 0:
        for d in debug_img_dir:
            shutil.rmtree(d)

    lc = QTable({'time': [150, 450] * u.s,
                'fluence_D1':  [4, 6] * (u.ph / u.cm**2 / u.s),
                'fluence_D2': [3, 1] * (u.ph / u.cm**2 / u.s)})
    lc_out = lightcurve_through_image(lc, 300 * u.s,
                                      final_resolution=600 * u.s)
    debug_img_dir = glob.glob('debug_imgs_*')
    assert len(debug_img_dir) == 0
    lc_out = lightcurve_through_image(lc, 300 * u.s,
                                      final_resolution=600 * u.s,
                                      debug=True)
    debug_img_dir = glob.glob('debug_imgs_*')
    assert len(debug_img_dir) > 0
    assert os.path.isdir(debug_img_dir[0])
    assert len(glob.glob(os.path.join(debug_img_dir[0], '*.p'))) > 0
    assert 'fluence_D1_fiterr' in lc_out.colnames
    assert len(lc_out['time']) == 1

    shutil.rmtree(debug_img_dir[0])


def test_numerically_realistic_lightcurve():
    import numpy as np
    ntrial=10

    ref_fluence = 10. * (u.ph / u.cm**2 / u.s)
    ref_lc = ref_fluence * np.ones(ntrial)

    lc = QTable({'time': np.arange(len(ref_lc)) * 300 * u.s,
                'fluence_D1': ref_lc,
                'fluence_D2': ref_lc})
    lc_out = lightcurve_through_image(lc, 300 * u.s,
                                      final_resolution=600 * u.s,
                                      frame=np.array([30, 30]),
                                      gal_type="elliptical")

    fl1_out = lc_out['fluence_D1_fit'].value
    fl2_out = lc_out['fluence_D2_fit'].value

    assert np.allclose(np.mean(fl1_out), ref_fluence.value, rtol=0.1)
    assert np.allclose(np.mean(fl2_out), ref_fluence.value, rtol=0.1)
