from eclipsoid.light_curve import limb_dark_oblate_lightcurve
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit
import jax.numpy as jnp
import pytest

@pytest.mark.parametrize(("params", "t"), [({
    'period':300.456,
    'radius':0.1,
    'u':jnp.array([0.3,0.2, 0.1, 0.1]),
    'f':0.,
    'bo':0.8,
    'duration':0.4,
    'theta':0.
}, jnp.linspace(-0.5, 0.5, 1000))])
def test_eclipsoid_spherical(params, t):
    orbit = TransitOrbit(
    period=params['period'], time_transit=0., duration=params['duration'], impact_param=params['bo'], radius_ratio=params['radius']
    )
    #add one since exoplanet/jaxoplanet convention is to return 0 as the mean
    lc = limb_dark_light_curve(orbit, params['u'])(t)+1.
    
    oblate_lc = limb_dark_oblate_lightcurve(orbit, params['u'], params['f'], params['theta'])(t)
    assert jnp.allclose(lc, oblate_lc, rtol=1e-7)
    
    