from astroduet.models import Simulations
import astropy.units as u


def test_sims():
    # Just test that it runs ok
    sims = Simulations()
    sims.parse_emgw(diag=True, list_of_simulations=['kilonova_0.01.dat'])
    sims.parse_sne(diag=True, list_of_simulations=['IIP_big'])
    sims.parse_sne_rsg(diag=True, list_of_simulations=['rsg400.dat'])
    sims.parse_sne_bsg(diag=True, list_of_simulations=['bsg20.dat'])
    sims.parse_sne_ysg(diag=True, list_of_simulations=['ysg150.dat'])