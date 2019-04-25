import os
import glob
import shutil
from astroduet.lightcurve import get_lightcurve, lightcurve_through_image
from astropy.table import Table
import astropy.units as u

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, '..', 'data')


def test_load_lightcurve():
    # Dummy test
    lc = get_lightcurve("shock_2.5e10.dat")
    assert 'time' in lc.colnames


def test_realistic_lightcurve():
    # Dummy test
    debug_img_dir = glob.glob('debug_imgs_*')
    if len(debug_img_dir) > 0:
        for d in debug_img_dir:
            shutil.rmtree(d)

    lc = Table({'time': [150, 450] * u.s,
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
