from astroduet.config import Telescope
import astropy.units as u


def test_run_through_Telescope_methods():
    duet = Telescope()
    duet_offax = Telescope(config='reduced_baseline')
    duet.info()
    duet.update_bandpass()
    duet.calc_radial_profile()
    duet.calc_psf_fwhm()
    duet.update_effarea()
    duet.fluence_to_rate(1 * u.ph / u.s / u.cm**2)
    duet.psf_model()
    duet.compute_psf_norms()


