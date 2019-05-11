from astroduet.models import Simulations
import astropy.units as u


def test_sims():
    # Just test that it runs ok
    sims = Simulations()
    sims.parse_emgw(diag=True)